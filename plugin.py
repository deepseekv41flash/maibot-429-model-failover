"""429 限流熔断与模型自动切换插件。

背景
----
MaiBot 在模型返回 429（限流）时会按 API Provider 的 ``max_retry`` 反复重试同一
个模型，重试间隔内依旧撞在限流窗口上，重试耗尽后才切换到下一个备选模型；而下
一次请求又会从列表第一个模型开始，于是很容易陷进「429 → 重试 → 429」的循环。

本插件做的事
------------
1. 监听宿主自己产出的 LLM 失败快照与 ``llm_models`` 日志，识别 429（可扩展到
   5xx 等）事件，拿到 ``任务（模型类别）`` 与 ``模型``；
2. 立刻把该模型从 ``config/model_config.toml`` 里对应任务的 ``model_list``
   中**暂时移除**——宿主 ``watchfiles`` 会在约 1 秒内热重载，于是下一个请求直接
   落到该模型类别里的下一个备选模型上；
3. 冻结时长可在插件设置页调整（默认 180 秒，可开启重复触发翻倍）；
4. 到期后按原位置把模型插回列表，配置恢复原样；插件卸载/停用同样会还原。

宿主没有对外开放「模型调度」相关的 Hook 或能力，因此这里是通过改写宿主模型
配置实现的；所有写入都先备份、做 TOML 解析与逐节比对、再原子替换，并且提供
演练模式（``dry_run``）可以先观察不落盘。详见 README。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

import asyncio
import contextlib
import time

from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ToolParamType, ToolParameterInfo

from .failover import (
    FailureDetector,
    FreezeManager,
    HostModelConfig,
    StateStore,
    build_settings,
)
from .failover import host_paths

PLUGIN_DISPLAY_NAME = "429 限流熔断"

RELOAD_CONFIRM_TIMEOUT_SECONDS = 12.0
"""写入模型配置后，等待宿主热重载确认的超时时间（秒）。"""

STARTUP_GRACE_SECONDS = 180.0
"""插件刚加载后的「启动宽限期」（秒）：这段时间内宿主可能还在启动，配置热重载广播会被丢弃，
收不到回执属于正常现象；宽限期内按 RELOAD_RETRY_INTERVAL_SECONDS 间隔反复重写配置触发重载，直到确认。"""

RELOAD_RETRY_INTERVAL_SECONDS = 30.0
"""启动宽限期内两次「重写配置以触发热重载」之间的间隔（秒）。"""


class PluginSectionConfig(PluginConfigBase):
    """基础设置。"""

    __ui_label__ = "基础设置"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        description="是否启用自动检测与冻结；关闭后只保留「到期还原」能力。",
        json_schema_extra={"label": "启用插件"},
    )
    admin_user_ids: List[str] = Field(
        default_factory=list,
        description="允许执行「手动冻结 / 解冻」命令的用户 ID，留空时仅本地 operator 可用。",
        json_schema_extra={"label": "管理员用户 ID", "placeholder": "例如 10001,10002"},
    )
    config_version: str = Field(
        default="1.2.3",
        description="配置版本（请勿手改）。",
        json_schema_extra={"label": "配置版本", "disabled": True},
    )


class DetectSectionConfig(PluginConfigBase):
    """检测设置。"""

    __ui_label__ = "检测设置"
    __ui_icon__ = "search"
    __ui_order__ = 1

    poll_interval_seconds: float = Field(
        default=3.0,
        ge=0.5,
        le=60.0,
        description="轮询宿主失败快照与日志的间隔（秒）。越短越及时，越小越费 IO。",
        json_schema_extra={"label": "检测间隔（秒）", "min": 0.5, "max": 60, "step": 0.5},
    )
    trigger_status_codes: List[int] = Field(
        default_factory=lambda: [429],
        description="触发冻结的 HTTP 状态码。429=限流；如需把 500/502/503/504 也纳入熔断可一并填入。",
        json_schema_extra={"label": "触发状态码", "placeholder": "429"},
    )
    detect_rate_limit_by_type: bool = Field(
        default=True,
        description="状态码缺失时，若异常类型或文本是限流特征（RateLimitError / 请求过于频繁等）同样触发。",
        json_schema_extra={"label": "按限流特征兜底判定"},
    )
    enable_snapshot_scan: bool = Field(
        default=True,
        description="扫描宿主 LLM 失败快照（logs/maisaka_prompt/llm_error），带结构化状态码，最可靠。",
        json_schema_extra={"label": "扫描失败快照"},
    )
    enable_log_scan: bool = Field(
        default=True,
        description="扫描宿主应用日志 app_*.log.jsonl 作为兜底通道。",
        json_schema_extra={"label": "扫描应用日志"},
    )
    scan_window_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=600.0,
        description="每轮最多回溯的时间窗口（秒），用于插件刚启用时避免读取过老的历史记录。",
        json_schema_extra={"label": "回溯窗口（秒）", "min": 5, "max": 600, "step": 5},
    )
    max_log_age_seconds: float = Field(
        default=300.0,
        ge=0.0,
        le=86400.0,
        description="日志通道只接受这个年龄以内的事件（秒），避免重启后把几百行以前的历史 429 当成刚发生的事；0 表示不限制。",
        json_schema_extra={"label": "日志事件最大年龄（秒）", "min": 0, "max": 86400, "step": 10},
    )


class ModelFreezeRuleConfig(PluginConfigBase):
    """单条「按模型冻结时长」规则。"""

    model: str = Field(
        default="",
        description="模型名，支持通配符：如 Qwen/Qwen3.7-Flash、inclusionai/*:free、*daily*。",
        json_schema_extra={"label": "模型（支持 * 通配）", "placeholder": "inclusionai/ling-3.1-flash:free"},
    )
    task: str = Field(
        default="",
        description="可选：限定任务/模型类别（replyer、planner、utils…）；留空表示所有任务。",
        json_schema_extra={"label": "限定任务（可留空）", "placeholder": "utils"},
    )
    duration: str = Field(
        default="5h",
        description="命中后的冻结时长，可读写法：30m / 5h / 1d / 1d12h / 1800（纯数字=秒）。",
        json_schema_extra={"label": "冻结时长", "placeholder": "5h"},
    )
    escalate: Literal["inherit", "on", "off"] = Field(
        default="inherit",
        description="重复触发时是否翻倍：inherit=跟随全局设置；off=固定为这里的时长（适合按天/按 5 小时限流的模型）。",
        json_schema_extra={"label": "重复触发翻倍"},
    )
    max_duration: str = Field(
        default="",
        description="可选：该规则的冻结时长上限（如 1d）；留空跟随全局上限。",
        json_schema_extra={"label": "时长上限（可留空）", "placeholder": "1d"},
    )
    window_duration: str = Field(
        default="",
        description="可选：该规则的「重复触发」判定窗口（如 12h）；留空跟随全局窗口。",
        json_schema_extra={"label": "倍增判定窗口（可留空）", "placeholder": "12h"},
    )
    note: str = Field(
        default="",
        description="备注，只用于自己辨识这条规则。",
        json_schema_extra={"label": "备注", "placeholder": "openrouter 免费档按天限流"},
    )


class FreezeSectionConfig(PluginConfigBase):
    """冻结策略。"""

    __ui_label__ = "冻结策略"
    __ui_icon__ = "snowflake"
    __ui_order__ = 2

    freeze_seconds: int = Field(
        default=180,
        ge=5,
        le=86400,
        description="被限流的模型「暂时冻结」多久（秒）。到期后自动放回备选列表。",
        json_schema_extra={"label": "冻结时长（秒）", "min": 5, "max": 86400},
    )
    escalate: bool = Field(
        default=True,
        description="同一模型在短时间内再次被限流时，冻结时长按 2 倍递增（上限见下一项）。",
        json_schema_extra={"label": "重复触发时长翻倍"},
    )
    max_freeze_seconds: int = Field(
        default=1800,
        ge=5,
        le=604800,
        description="翻倍升级后的冻结时长上限（秒）。",
        json_schema_extra={"label": "冻结时长上限（秒）", "min": 5, "max": 604800},
    )
    escalation_window_seconds: int = Field(
        default=900,
        ge=0,
        le=86400,
        description="多久之内再次被限流算作「重复触发」。0 表示每次都重新计时。",
        json_schema_extra={"label": "重复触发判定窗口（秒）", "min": 0, "max": 86400},
    )
    model_rules: List[ModelFreezeRuleConfig] = Field(
        default_factory=list,
        description=(
            "按模型单独设置冻结时长：不同上游的限流窗口不一样（有的按分钟、有的按 5 小时、有的按天），"
            "这里可以逐条指定。规则从上往下匹配，第一条命中即生效，因此把更具体的写在前面。"
            "没有命中的模型使用上面的全局「冻结时长（秒）」。"
        ),
        json_schema_extra={"label": "按模型自定义冻结时长", "hint": "示例：Qwen/Qwen3.7-Flash → 5h；inclusionai/*:free → 1d"},
    )
    freeze_scope: Literal["task", "model"] = Field(
        default="task",
        description="task=只冻结出问题的那个模型类别（推荐）；model=该模型出现的所有类别都暂时移除。",
        json_schema_extra={"label": "冻结范围"},
    )
    exclude_tasks: List[str] = Field(
        default_factory=lambda: ["embedding", "image_embedding"],
        description="不参与冻结的任务（模型类别）。嵌入类切换会污染向量库，默认排除。",
        json_schema_extra={"label": "排除的任务", "placeholder": "embedding,image_embedding"},
    )
    exclude_models: List[str] = Field(
        default_factory=list,
        description="不参与冻结的模型，支持通配符，例如 local/* 。",
        json_schema_extra={"label": "排除的模型", "placeholder": "local/*"},
    )
    min_alternatives: int = Field(
        default=1,
        ge=0,
        le=10,
        description="移除后该类别至少还要保留几个模型，不足时跳过冻结（避免把列表清空导致请求直接失败）。",
        json_schema_extra={"label": "剩余模型保底数量", "min": 0, "max": 10},
    )
    skip_if_no_valid_alternative: bool = Field(
        default=True,
        description="若该类别剩下的候选模型都没有在 [[models]] 里声明（调用必然报「未找到名为…的模型」），则跳过冻结，避免把流量推向无效模型。",
        json_schema_extra={"label": "无有效备选时跳过冻结"},
    )
    dry_run: bool = Field(
        default=False,
        description="演练模式：只记录与通知，不改写宿主机型配置。建议首次使用时先开一段时间观察。",
        json_schema_extra={"label": "演练模式（不改写配置）"},
    )
    restore_on_unload: bool = Field(
        default=True,
        description="插件卸载或停用时，是否立刻还原所有仍处于冻结中的模型。",
        json_schema_extra={"label": "卸载时还原"},
    )
    reassert_on_external_change: bool = Field(
        default=True,
        description="有人在 WebUI／手动改配置把冻结中的模型放回列表时，是否重新移除它。",
        json_schema_extra={"label": "外部改动后重新施加冻结"},
    )
    respect_manual_edit: bool = Field(
        default=True,
        description=(
            "冻结期间有人改过该任务的 model_list 且列表里已不含被冻结的模型时，"
            "到期不再把它插回去（尊重手工编辑，避免覆盖你的改动）；关掉则无论列表被怎么改都还原。"
        ),
        json_schema_extra={"label": "尊重冻结期间的手工改动"},
    )


class NotifySectionConfig(PluginConfigBase):
    """通知设置。"""

    __ui_label__ = "通知设置"
    __ui_icon__ = "bell"
    __ui_order__ = 3

    enabled: bool = Field(
        default=False,
        description="冻结/还原时向指定聊天流发送一条提示消息。",
        json_schema_extra={"label": "启用通知"},
    )
    stream_id: str = Field(
        default="",
        description="接收通知的聊天流 ID（例如 qq_group_123456），留空则不发送。",
        json_schema_extra={"label": "通知聊天流", "placeholder": "qq_group_123456"},
    )
    notify_on_freeze: bool = Field(
        default=True,
        description="冻结模型时发送通知。",
        json_schema_extra={"label": "冻结时通知"},
    )
    notify_on_unfreeze: bool = Field(
        default=True,
        description="还原模型时发送通知。",
        json_schema_extra={"label": "还原时通知"},
    )


class AdvancedSectionConfig(PluginConfigBase):
    """高级设置。"""

    __ui_label__ = "高级设置"
    __ui_icon__ = "settings-2"
    __ui_order__ = 4

    maibot_root: str = Field(
        default="",
        description="MaiBot 根目录；留空自动探测（依据 MAIBOT_PLUGIN_DIRS / 工作目录 / 插件自身位置）。",
        json_schema_extra={"label": "MaiBot 根目录", "placeholder": "/MaiMBot"},
    )
    model_config_path: str = Field(
        default="",
        description="模型配置文件路径；留空使用 <根目录>/config/model_config.toml。",
        json_schema_extra={"label": "模型配置文件路径", "placeholder": "config/model_config.toml"},
    )
    log_dir: str = Field(
        default="",
        description="宿主日志目录；留空使用 <根目录>/logs。",
        json_schema_extra={"label": "日志目录", "placeholder": "logs"},
    )
    backup_keep: int = Field(
        default=10,
        ge=0,
        le=200,
        description="每次改写前备份模型配置，这里保留最近几份。",
        json_schema_extra={"label": "保留备份份数", "min": 0, "max": 200},
    )


class ModelFailoverConfig(PluginConfigBase):
    """插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    detect: DetectSectionConfig = Field(default_factory=DetectSectionConfig)
    freeze: FreezeSectionConfig = Field(default_factory=FreezeSectionConfig)
    notify: NotifySectionConfig = Field(default_factory=NotifySectionConfig)
    advanced: AdvancedSectionConfig = Field(default_factory=AdvancedSectionConfig)


class ModelFailoverPlugin(MaiBotPlugin):
    """429 限流熔断与模型自动切换。"""

    config_model = ModelFailoverConfig
    config_reload_subscriptions: ClassVar[Tuple[str, ...]] = ("model",)
    """宿主模型配置热重载时收到通知，可用于重新施加冻结。"""

    def __init__(self) -> None:
        """初始化插件状态。"""

        super().__init__()
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._closing = False
        self._manager: Optional[FreezeManager] = None
        self._detector: Optional[FailureDetector] = None
        self._store: Optional[StateStore] = None
        self._host_config: Optional[HostModelConfig] = None
        self._data_dir: Optional[Path] = None
        self._self_write_at: float = 0.0
        self._reassert_task: Optional[asyncio.Task[None]] = None
        self._awaiting_reload_at: float = 0.0
        self._last_reload_at: float = 0.0
        self._reload_timeout_warned = False
        self._health_checked_at: float = 0.0
        self._health_signature: str = ""
        self._loaded_at: float = 0.0
        self._startup_retry_count: int = 0
        self._reload_retry_at: float = 0.0

    # ------------------------------------------------------------------ 生命周期

    async def on_load(self) -> None:
        """插件加载：解析路径、恢复历史状态、启动检测循环。"""

        settings = build_settings(self.config)
        self._closing = False
        data_dir = Path(self.ctx.paths.data_dir)
        self._data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        self._store = StateStore(data_dir / "state.json", self.ctx.logger)
        self._store.load()

        model_config_path = host_paths.resolve_model_config_path(
            settings.maibot_root,
            settings.model_config_path,
        )
        if model_config_path is None:
            self.ctx.logger.error(
                "无法定位宿主 model_config.toml，插件将只有检测能力。"
                "请在插件设置 → 高级设置里填写 MaiBot 根目录或模型配置文件路径。"
            )
            model_config_path = Path("model_config.toml")

        self._host_config = HostModelConfig(model_config_path, self.ctx.logger)
        self._manager = FreezeManager(
            host_config=self._host_config,
            store=self._store,
            logger=self.ctx.logger,
            settings_provider=lambda: build_settings(self.config),
            backup_dir=data_dir / "backups",
            notifier=self._send_notice,
            on_write=self._mark_self_write,
        )
        self._detector = FailureDetector(
            logger=self.ctx.logger,
            error_dir=host_paths.resolve_llm_error_dir(settings.maibot_root, settings.log_dir),
            log_dir=host_paths.resolve_log_dir(settings.maibot_root, settings.log_dir),
            status_codes=settings.status_codes,
            detect_rate_limit_by_type=settings.detect_rate_limit_by_type,
            enable_snapshot_scan=settings.enable_snapshot_scan,
            enable_log_scan=settings.enable_log_scan,
            window_seconds=settings.scan_window_seconds,
            max_log_event_age_seconds=settings.max_log_age_seconds,
        )

        self.ctx.logger.info(
            f"{PLUGIN_DISPLAY_NAME} 已加载：模型配置 {self._host_config.path}；"
            f"{self._detector.describe_sources()}；"
            f"{'演练模式' if settings.dry_run else '实写模式'}"
        )

        restored = await self._manager.restore_all(reason="插件加载时还原上次残留")
        if restored:
            self.ctx.logger.info(f"启动时还原残留冻结：{'；'.join(restored)}")

        self._loaded_at = time.time()
        self._startup_retry_count = 0
        self._reload_retry_at = 0.0
        self._loop_task = asyncio.create_task(self._run_loop(), name="429-model-failover-loop")
        self._store.save()

    async def on_unload(self) -> None:
        """插件卸载：停止循环并按配置还原冻结。"""

        self._closing = True
        for task in (self._loop_task, self._reassert_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._loop_task = None
        self._reassert_task = None

        settings = build_settings(self.config)
        if self._manager is not None and settings.restore_on_unload:
            with contextlib.suppress(Exception):
                await self._manager.restore_all(reason="插件卸载")

        if self._store is not None:
            self._store.save()
        self.ctx.logger.info(f"{PLUGIN_DISPLAY_NAME} 已卸载")

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        """配置热重载。

        ``scope == "model"`` 表示宿主模型配置发生变化：可能是本插件刚写完配置后
        宿主热重载的回执，也可能是 WebUI／手工编辑。前者用于确认「配置改动真的
        生效了」，后者则要检查冻结中的模型有没有被放回列表。
        """

        del config_data, version
        if scope != "model":
            return

        now = time.time()
        if self._awaiting_reload_at:
            # 迟到的确认也算确认：宿主事件循环可能卡顿十几秒（日志里出现过 15s 的
            # 迟到告警），不能因为超过阈值就把配置改动误判成「没生效」。
            waited = now - self._awaiting_reload_at
            was_late = waited > RELOAD_CONFIRM_TIMEOUT_SECONDS
            self._awaiting_reload_at = 0.0
            self._last_reload_at = now
            self._reload_timeout_warned = False
            self.ctx.logger.info(
                f"宿主已{'（延迟 %.0f 秒后）' % waited if was_late else ''}确认热重载模型配置，冻结/还原改动已生效"
            )
            return

        if now - self._self_write_at < 5.0:
            return
        if self._manager is None:
            return
        if self._reassert_task is not None and not self._reassert_task.done():
            return
        self._reassert_task = asyncio.create_task(self._reassert_after_edit())

    # ------------------------------------------------------------------ 主循环

    async def _run_loop(self) -> None:
        """后台检测循环。"""

        while not self._closing:
            interval = RELOAD_CONFIRM_TIMEOUT_SECONDS
            try:
                # 配置读取也放在 try 内：任何异常都不能让后台任务静默退出
                settings = build_settings(self.config)
                interval = max(0.5, settings.poll_interval_seconds)
                if settings.enabled:
                    self._refresh_detector(settings)
                    detections = await asyncio.to_thread(self._detector.scan) if self._detector else []
                    fresh = []
                    for detection in detections:
                        if self._store is not None and not self._store.mark_processed(detection.event_id):
                            continue
                        self.ctx.logger.info(f"检测到限流事件：{detection.describe()}")
                        fresh.append(detection)
                    if fresh and self._manager is not None:
                        await self._manager.process(fresh)
                if self._manager is not None:
                    await self._manager.expire_due()
                self._check_reload_confirmation()
                self._check_config_health()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 循环必须保持存活
                self.ctx.logger.error(f"检测循环出现异常：{exc}", exc_info=True)

            await asyncio.sleep(max(0.5, interval))

    def _check_reload_confirmation(self) -> None:
        """检查宿主是否确认了本插件刚写入的模型配置。

        宿主的文件监听挂在 inode 上：如果 ``model_config.toml`` 被别的东西用
        「换 inode」的方式写过（宿主 WebUI 保存、编辑器原子保存等），监听可能静默
        失效，此后本插件的改动不会被宿主读取。这里给出明确告警，而不是让用户
        以为插件在正常工作。
        """

        if not self._awaiting_reload_at or self._reload_timeout_warned:
            return
        now = time.time()
        waited = now - self._awaiting_reload_at
        if waited < RELOAD_CONFIRM_TIMEOUT_SECONDS:
            return

        # 启动宽限期内：宿主可能仍在启动（此时它的配置热重载广播会被直接丢弃），
        # 所以不按「监听失效」告警，而是按固定间隔反复就地重写配置触发重载，直到收到确认。
        if self._loaded_at and now - self._loaded_at <= STARTUP_GRACE_SECONDS:
            if now < self._reload_retry_at:
                return
            if self._manager is None or not self._manager.touch_config():
                return
            self._awaiting_reload_at = now  # 重新开始等待确认（写入回调通常已刷新，这里兜底）
            self._startup_retry_count += 1
            self._reload_retry_at = now + RELOAD_RETRY_INTERVAL_SECONDS
            if self._startup_retry_count == 1:
                self.ctx.logger.warning(
                    f"插件刚加载 {int(now - self._loaded_at)} 秒，写入的配置未收到宿主热重载回执"
                    f"（宿主可能仍在启动）；将每 {int(RELOAD_RETRY_INTERVAL_SECONDS)} 秒就地重写一次配置以触发热重载，"
                    "直到收到确认为止"
                )
            else:
                self.ctx.logger.debug(
                    f"再次就地重写宿主模型配置以触发热重载（第 {self._startup_retry_count} 次，"
                    f"插件加载后 {int(now - self._loaded_at)} 秒）"
                )
            return

        self._reload_timeout_warned = True
        self.ctx.logger.error(
            f"写入模型配置后 {int(waited)} 秒仍未收到宿主热重载确认，"
            "配置改动可能没有生效（常见原因：宿主的文件监听失效，例如 model_config.toml "
            "被换 inode 的写入方式覆盖过）。建议重启一次 MaiBot。"
        )

    def _check_config_health(self, interval_seconds: float = 600.0) -> None:
        """定期做一次配置体检：任务列表里引用了未声明模型时给出明确告警。

        这类问题不会产生 429，插件的熔断逻辑帮不上忙，但会让相关任务每次都直接
        失败（宿主在罚分之前就抛错，``balance`` 策略还会反复挑中它），因此值得
        主动提示用户去补声明或删掉名字。
        """

        if self._manager is None:
            return
        now = time.time()
        if now - self._health_checked_at < interval_seconds:
            return
        self._health_checked_at = now

        lines = self._manager.health_lines()
        signature = "\n".join(lines)
        if not lines:
            self._health_signature = ""
            return
        if signature == self._health_signature:
            return
        self._health_signature = signature
        for line in lines:
            self.ctx.logger.warning(f"配置体检：{line}")

    def _health_status_lines(self) -> list[str]:
        """把配置体检结果并入状态输出。"""

        if self._manager is None:
            return []
        return self._manager.health_lines()

    def _refresh_detector(self, settings: Any) -> None:
        """配置热重载后同步检测器参数。"""

        if self._detector is None:
            return
        self._detector.status_codes = set(settings.status_codes)
        self._detector.detect_rate_limit_by_type = settings.detect_rate_limit_by_type
        self._detector.enable_snapshot_scan = settings.enable_snapshot_scan
        self._detector.enable_log_scan = settings.enable_log_scan
        self._detector.window_seconds = settings.scan_window_seconds
        self._detector.max_log_event_age_seconds = settings.max_log_age_seconds

    async def _reassert_after_edit(self) -> None:
        """稍等宿主热重载完成后重新施加冻结。"""

        try:
            await asyncio.sleep(3.0)
            if self._manager is not None:
                await self._manager.reassert()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.warning(f"重新施加冻结失败：{exc}")

    async def _send_notice(self, text: str) -> None:
        """向配置的聊天流发送通知。"""

        stream_id = build_settings(self.config).notify_stream_id
        if not stream_id:
            return
        await self.ctx.send.text(f"[{PLUGIN_DISPLAY_NAME}]\n{text}", stream_id)

    def _mark_self_write(self) -> None:
        """记录一次由本插件发起的配置写入，并开始等待宿主热重载确认。"""

        self._self_write_at = time.time()
        self._awaiting_reload_at = self._self_write_at
        self._reload_timeout_warned = False
        self._reload_retry_at = 0.0

    def _reload_status_line(self) -> str:
        """生成「宿主是否确认热重载」的状态行。

        注意先判超时告警再判「等待中」：超时后 ``_awaiting_reload_at`` 会一直保留
        （等着迟到的确认来清除），如果先判等待中，告警分支就永远显示不出来。
        """

        if self._reload_timeout_warned:
            waited = int(time.time() - self._awaiting_reload_at) if self._awaiting_reload_at else 0
            return f"宿主热重载确认：⚠ 已等待 {waited} 秒仍未确认，配置改动可能未生效，建议重启 MaiBot"
        if self._awaiting_reload_at:
            waited = int(time.time() - self._awaiting_reload_at)
            return f"宿主热重载确认：等待中（已 {waited} 秒）"
        if self._last_reload_at:
            ago = int(time.time() - self._last_reload_at)
            return f"宿主热重载确认：正常（{ago} 秒前确认过一次）"
        return "宿主热重载确认：尚未写入过配置"

    # ------------------------------------------------------------------ 命令

    @Command(
        "failover_status",
        description="查看 429 限流熔断插件的冻结状态",
        pattern=r"^/(?:429|429状态|429status)\s*$",
    )
    async def show_status(self, stream_id: str = "", **kwargs: Any) -> Tuple[bool, str, int]:
        """输出当前冻结状态。"""

        if self._manager is None:
            return False, "插件尚未完成初始化", 2

        # 本命令对所有人开放，因此宿主侧的绝对路径只回显给管理员：
        # 否则群里任何人发一句 /429 就能拿到服务器的目录布局。
        settings = build_settings(self.config)
        is_admin = settings.is_admin(
            str(kwargs.get("user_id", "")),
            bool(kwargs.get("is_local_operator")),
        )

        text = "\n".join(
            [f"{PLUGIN_DISPLAY_NAME} 状态"]
            + self._manager.status_lines()
            + self._health_status_lines()
            + [self._reload_status_line()]
        )
        if self._host_config is not None:
            target = (
                str(self._host_config.path)
                if is_admin
                else f"{self._host_config.path.name}（宿主路径仅管理员可见）"
            )
            text += f"\n模型配置：{target}"
        if self._detector is not None:
            text += f"\n数据来源：{self._detector.describe_sources(show_paths=is_admin)}"
        await self.ctx.send.text(text, stream_id)
        return True, text, 2

    @Command(
        "failover_unfreeze",
        description="手动解冻模型；/429解冻 全部，或 /429解冻 <任务> [模型]",
        pattern=r"^/(?:429解冻|429unfreeze)(?:\s+(?P<task>\S+))?(?:\s+(?P<model>\S+))?\s*$",
    )
    async def unfreeze(self, stream_id: str = "", **kwargs: Any) -> Tuple[bool, str, int]:
        """手动解冻。"""

        settings = build_settings(self.config)
        if not settings.is_admin(
            str(kwargs.get("user_id", "")),
            bool(kwargs.get("is_local_operator")),
        ):
            return False, "只有插件管理员可以执行解冻", 2
        if self._manager is None:
            return False, "插件尚未完成初始化", 2

        groups = kwargs.get("matched_groups") or {}
        task = str(groups.get("task") or "").strip()
        model = str(groups.get("model") or "").strip()
        if task.lower() in {"all", "全部", "*"}:
            task = ""
        count, messages = await self._manager.unfreeze(task, model)
        text = "\n".join(messages) if messages else f"已解冻 {count} 条记录"
        await self.ctx.send.text(text, stream_id)
        return True, text, 2

    @Command(
        "failover_freeze",
        description="手动冻结模型：/429冻结 <任务> <模型> [秒]",
        pattern=r"^/(?:429冻结|429freeze)\s+(?P<task>\S+)\s+(?P<model>\S+)(?:\s+(?P<seconds>\d+))?\s*$",
    )
    async def freeze_now(self, stream_id: str = "", **kwargs: Any) -> Tuple[bool, str, int]:
        """手动冻结。"""

        settings = build_settings(self.config)
        if not settings.is_admin(
            str(kwargs.get("user_id", "")),
            bool(kwargs.get("is_local_operator")),
        ):
            return False, "只有插件管理员可以执行手动冻结", 2
        if self._manager is None:
            return False, "插件尚未完成初始化", 2

        groups = kwargs.get("matched_groups") or {}
        task = str(groups.get("task") or "").strip()
        model = str(groups.get("model") or "").strip()
        raw_seconds = str(groups.get("seconds") or "").strip()
        seconds = int(raw_seconds) if raw_seconds.isdigit() else 0
        text = await self._manager.manual_freeze(task, model, seconds)
        await self.ctx.send.text(text, stream_id)
        return True, text, 2

    @Command(
        "failover_help",
        description="429 限流熔断插件帮助",
        pattern=r"^/(?:429帮助|429help)\s*$",
    )
    async def show_help(self, stream_id: str = "", **kwargs: Any) -> Tuple[bool, str, int]:
        """输出帮助。"""

        del kwargs
        text = (
            f"{PLUGIN_DISPLAY_NAME} 帮助\n"
            "/429 或 /429状态 查看冻结中的模型与剩余时间\n"
            "/429解冻 全部 | /429解冻 <任务> | /429解冻 <任务> <模型>\n"
            "/429冻结 <任务> <模型> [秒]\n"
            "任务名即 model_config.toml 中 [model_task_config.<任务>] 的名字，"
            "例如 replyer、planner、utils、fast_model。\n"
            "插件会自动检测 429 并把出问题的模型从该任务的备选列表里暂时移除，"
            "冻结到期后自动放回。"
        )
        await self.ctx.send.text(text, stream_id)
        return True, text, 2

    # ------------------------------------------------------------------ 工具

    @Tool(
        "model_freeze_status",
        description="查询当前被限流冻结的模型、剩余冻结时间与最近的限流事件；排查模型报错时使用。",
        parameters=[],
    )
    async def tool_status(self, **kwargs: Any) -> Dict[str, str]:
        """给 LLM 的状态查询工具。"""

        del kwargs
        if self._manager is None:
            return {"content": "429 熔断插件尚未完成初始化"}
        lines = [*self._manager.status_lines(), *self._health_status_lines(), self._reload_status_line()]
        return {"content": "\n".join(lines)}

    @Tool(
        "unfreeze_model",
        description="把被 429 限流冻结的模型立即放回备选列表；留空参数表示解冻全部。",
        parameters=[
            ToolParameterInfo(
                name="task",
                param_type=ToolParamType.STRING,
                description="任务/模型类别名，例如 replyer、utils；留空表示所有任务",
                required=False,
            ),
            ToolParameterInfo(
                name="model",
                param_type=ToolParamType.STRING,
                description="模型名；留空表示该任务下所有被冻结的模型",
                required=False,
            ),
        ],
    )
    async def tool_unfreeze(self, task: str = "", model: str = "", **kwargs: Any) -> Dict[str, str]:
        """给 LLM 的解冻工具（与 ``/429解冻`` 同权限：仅插件管理员可用）。

        工具会被模型自主调用，因此这里必须和命令走同一道管理员校验 —— 否则任何
        能让机器人调用工具的聊天用户都能绕过 ``/429解冻`` 的权限限制去改写宿主配置。
        宿主会把 ``user_id`` / ``group_id`` / ``platform`` / ``stream_id`` 放进工具
        handler 的 kwargs（``component_query.py::_build_tool_context_payload``）。
        """

        settings = build_settings(self.config)
        if not settings.is_admin(
            str(kwargs.get("user_id", "")),
            bool(kwargs.get("is_local_operator")),
        ):
            return {"content": "只有插件管理员可以解冻模型（与 /429解冻 同权限）；请让管理员执行。"}
        if self._manager is None:
            return {"content": "429 熔断插件尚未完成初始化"}
        _count, messages = await self._manager.unfreeze(str(task).strip(), str(model).strip())
        return {"content": "\n".join(messages) if messages else "没有匹配的冻结记录"}


def create_plugin() -> ModelFailoverPlugin:
    """宿主固定按这个名字查找工厂函数。"""

    return ModelFailoverPlugin()

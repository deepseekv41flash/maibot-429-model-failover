"""冻结管理器：把“检测到 429”翻译成“宿主机型列表改写 + 到期还原”。

工作模型
--------
MaiBot 的任务配置 ``[model_task_config.<任务>]`` 中的 ``model_list`` 就是“该模型
类别的备选模型列表”。冻结一个模型 = 把它从列表里暂时移除（原位插入需要的信息
记录在冻结条目里），于是：

* ``sequential`` 策略立刻改用列表里的下一个备选模型；
* ``balance`` / ``random`` 策略也不会再抽到这个模型；
* 冻结到期后按原位置插回，配置回到用户原本的样子。

安全约束
--------
* 每轮改写前都重新读取宿主配置，只替换目标任务的 ``model_list`` 数组区间；
* 移除后该任务至少要剩 ``min_alternatives`` 个模型（默认 1），否则跳过冻结，
  避免把任务列表清空导致请求直接失败；
* 嵌入类任务默认排除（向量空间不能混用）；
* 写盘前后做 TOML 解析与逐节比对，任何异常都不落盘；
* 卸载/关闭插件时按配置还原全部冻结。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import asyncio
import contextlib
import time

from .detector import DetectedRateLimit
from .host_config import ArrayStyle, HostModelConfig, ModelConfigError
from .rules import format_duration, unmatched_rules
from .settings import FailoverSettings
from .state import FreezeEntry, StateStore

BURST_DEDUPE_SECONDS = 90.0
"""同一模型在这段时间内重复上报，视为同一轮限流，不重复计数/不重复改写。"""

RESTORE_RETRY_SECONDS = 60.0
"""还原失败后的重试间隔。"""


@dataclass(frozen=True)
class _ModelPosition:
    """模型在某个任务列表里的位置（含前后锚点）。"""

    index: int
    after_model: str
    before_model: str


def _position_of(models: Sequence[str], model: str) -> _ModelPosition:
    """计算模型在列表中的位置与前后锚点。"""

    names = [str(item) for item in models]
    try:
        index = names.index(model)
    except ValueError:
        return _ModelPosition(index=0, after_model="", before_model="")
    return _ModelPosition(
        index=index,
        after_model=names[index - 1] if index > 0 else "",
        before_model=names[index + 1] if index + 1 < len(names) else "",
    )


class FreezeManager:
    """冻结/还原宿主机型列表的调度器。"""

    def __init__(
        self,
        *,
        host_config: HostModelConfig,
        store: StateStore,
        logger: Any,
        settings_provider: Callable[[], FailoverSettings],
        backup_dir: Optional[Path] = None,
        notifier: Optional[Callable[[str], Any]] = None,
        on_write: Optional[Callable[[], None]] = None,
    ) -> None:
        """初始化冻结管理器。

        Args:
            host_config: 宿主机型配置编辑器。
            store: 状态仓库。
            logger: 插件日志器。
            settings_provider: 返回当前配置快照的函数（支持热重载）。
            backup_dir: 配置备份目录。
            notifier: 可选的通知回调（接收文本）。
            on_write: 每次成功改写宿主配置后的回调（用于标记“这是本插件写的”）。
        """

        self._host = host_config
        self._store = store
        self._logger = logger
        self._settings_provider = settings_provider
        self._backup_dir = backup_dir
        self._notifier = notifier
        self._on_write = on_write
        self._lock = asyncio.Lock()
        self._restore_retry_at: Dict[str, float] = {}
        self._declared_models: set[str] = set()
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_cached_at: float = 0.0

    # ------------------------------------------------------------------ 只读

    @property
    def settings(self) -> FailoverSettings:
        """当前配置快照。"""

        return self._settings_provider()

    def frozen_entries(self, now: Optional[float] = None) -> List[FreezeEntry]:
        """返回当前仍处于冻结中的条目（按剩余时间升序）。"""

        current = time.time() if now is None else now
        entries = [entry for entry in self._store.state.freezes if entry.until > current]
        entries.sort(key=lambda item: item.until)
        return entries

    def status_lines(self, now: Optional[float] = None) -> List[str]:
        """生成给用户看的冻结状态文本行。"""

        settings = self.settings
        current = time.time() if now is None else now
        entries = self.frozen_entries(current)
        stats = self._store.state.stats
        lines: List[str] = []
        if entries:
            lines.append(f"当前冻结 {len(entries)} 个模型：")
            for entry in entries:
                scope = "全局" if entry.scope == "model" else "单任务"
                plan = settings.plan_for(entry.task, entry.model)
                dry = "（演练模式，未真正改写）" if not entry.applied else ""
                lines.append(
                    f"· [{entry.task}] {entry.model} → 剩余 {entry.describe_remaining(current)}"
                    f"（{scope}，第 {entry.repeat} 次，{plan.label}）{dry}"
                )
        else:
            lines.append("当前没有被冻结的模型。")
        lines.append(
            "累计：检测 {detected} 次，冻结 {frozen} 次，还原 {restored} 次，改写失败 {failed} 次".format(
                detected=int(stats.get("detected", 0) or 0),
                frozen=int(stats.get("frozen", 0) or 0),
                restored=int(stats.get("restored", 0) or 0),
                failed=int(stats.get("write_failed", 0) or 0),
            )
        )
        last = stats.get("last_event")
        if isinstance(last, dict) and last.get("summary"):
            lines.append(f"最近事件：{last.get('summary')}")
        mode = "演练模式（不改写）" if settings.dry_run else "实写模式"
        for rule_line in settings.plan_lines():
            lines.append(rule_line)
        lines.append(
            f"策略：默认冻结 {format_duration(settings.freeze_seconds)}"
            f"{'（重复触发翻倍，上限 ' + format_duration(settings.max_freeze_seconds) + '）' if settings.escalate else ''}"
            f"，范围 {'该模型的所有任务' if settings.global_scope else '仅出问题的任务'}，{mode}"
        )
        return lines

    # ------------------------------------------------------------------ 检测

    async def process(self, detections: Sequence[DetectedRateLimit]) -> List[str]:
        """处理一轮检测结果。

        Args:
            detections: 本轮检测到的限流事件。

        Returns:
            List[str]: 人类可读的处理说明（用于日志与通知）。
        """

        settings = self.settings
        if not settings.enabled:
            return []

        now = time.time()
        messages: List[str] = []
        pending: List[Tuple[DetectedRateLimit, str, _ModelPosition]] = []
        styles: Dict[str, ArrayStyle] = {}

        async with self._lock:
            working, original_text, declared, error = self._load_working_lists()
            if error is not None:
                self._logger.warning(error)
                return []
            self._declared_models = declared
            self._health_cache = None
            original = {task: list(models) for task, models in working.items()}

            if not settings.dry_run:
                # 演练模式的记录不会真的改写配置，因此不该占用冻结名额：
                # 关闭演练模式后立刻清掉它们，让真实冻结马上生效。
                for stale in [item for item in self._store.state.freezes if not item.applied]:
                    self._store.state.remove_freeze(stale.task, stale.model)

            for detection in self._dedupe_batch(detections):
                self._store.bump("detected")
                skip_reason = self._check_skippable(detection, settings, working, declared, now)
                if skip_reason:
                    self._logger.debug(
                        f"跳过限流事件 [{detection.task}] {detection.model}（{skip_reason}）"
                    )
                    messages.append(f"跳过 {detection.model}（{skip_reason}）")
                    continue
                for task in self._target_tasks(detection, settings, working):
                    current = working.get(task, [])
                    if detection.model not in current:
                        continue
                    if len(current) - 1 < settings.min_alternatives:
                        warning = (
                            f"跳过 [{task}] {detection.model}：移除后仅剩 {len(current) - 1} 个模型，"
                            f"低于保底 {settings.min_alternatives} 个"
                        )
                        self._logger.warning(warning)
                        messages.append(warning)
                        continue
                    # 位置信息一律按“本轮开始时的原始列表”计算，
                    # 否则批量冻结时后续模型的索引会随着列表缩短而错位。
                    position = _position_of(original.get(task, []), detection.model)
                    working[task] = [name for name in current if name != detection.model]
                    if task not in styles:
                        with contextlib.suppress(ModelConfigError):
                            styles[task] = self._host.array_style(original_text, task)
                    pending.append((detection, task, position))

            if not pending:
                self._persist()
                return messages

            updates = {
                task: models
                for task, models in working.items()
                if models != original.get(task)
            }
            try:
                result = self._host.apply(
                    updates,
                    backup_dir=self._backup_dir,
                    backup_keep=max(0, settings.backup_keep),
                    dry_run=settings.dry_run,
                )
            except ModelConfigError as exc:
                self._store.bump("write_failed")
                self._store.note(
                    "last_event",
                    {"summary": f"改写宿主模型配置失败：{exc}", "at": now},
                )
                self._logger.error(f"改写宿主模型配置失败，已放弃本轮冻结：{exc}")
                self._persist()
                return [f"改写宿主模型配置失败：{exc}"]

            if not result.changed:
                self._persist()
                return messages

            self._note_write(result.written)

            for detection, task, position in pending:
                repeat = self._next_repeat(task, detection.model, now, settings)
                plan = settings.plan_for(task, detection.model)
                entry = FreezeEntry(
                    task=task,
                    model=detection.model,
                    index=position.index,
                    array_style=styles.get(task, ArrayStyle()).to_dict(),
                    context_models=list(original.get(task, [])),
                    left_models=list(working.get(task, [])),
                    scope="model" if settings.global_scope else "task",
                    frozen_at=now,
                    until=now + plan.duration(repeat),
                    repeat=repeat,
                    reason=f"HTTP {detection.status_code}（{detection.source}）",
                    applied=result.written,
                    status_code=detection.status_code,
                    source=detection.source,
                    after_model=position.after_model,
                    before_model=position.before_model,
                )
                self._store.state.add_freeze(entry)
                self._store.bump("frozen")
                self._remember_history(task, detection.model, now, repeat)
                summary = (
                    f"任务 [{task}] 的模型 {detection.model} 触发 {detection.status_code or '限流'}，"
                    f"已冻结 {format_duration(plan.duration(repeat))}（{plan.label}）"
                    f"{'（演练模式未改写）' if not result.written else ''}"
                    f"，改用备选模型：{self._describe_alternatives(working.get(task, []), declared)}"
                )
                self._logger.warning(summary)
                messages.append(summary)

            if messages:
                self._store.note("last_event", {"summary": messages[-1], "at": now})
            self._persist()

        if settings.notify_enabled and settings.notify_on_freeze:
            await self._notify("\n".join(messages))
        return messages

    async def expire_due(self) -> List[str]:
        """还原所有到期的冻结。"""

        now = time.time()
        due = [
            entry
            for entry in self._store.state.freezes
            if entry.until <= now and self._restore_retry_at.get(entry.key, 0.0) <= now
        ]
        if not due:
            return []
        return await self._restore(due, reason="冻结到期")

    async def restore_all(self, reason: str = "插件卸载") -> List[str]:
        """还原全部冻结记录。"""

        entries = list(self._store.state.freezes)
        if not entries:
            return []
        return await self._restore(entries, reason=reason)

    async def unfreeze(self, task: str = "", model: str = "") -> Tuple[int, List[str]]:
        """手动解冻：留空表示全部解冻。

        Args:
            task: 目标任务（模型类别）。
            model: 目标模型。

        Returns:
            tuple[int, list[str]]: 命中条数与消息列表。
        """

        entries = list(self._store.state.freezes)
        if task:
            entries = [entry for entry in entries if entry.task == task]
        if model:
            entries = [entry for entry in entries if entry.model == model]
        if not entries:
            return 0, ["没有匹配的冻结记录。"]
        messages = await self._restore(entries, reason="手动解冻")
        return len(entries), messages

    async def manual_freeze(self, task: str, model: str, seconds: int = 0) -> str:
        """手动冻结一个模型。

        Args:
            task: 目标任务（模型类别）。
            model: 目标模型。
            seconds: 冻结秒数；``0`` 表示使用配置里的默认时长。

        Returns:
            str: 结果说明。
        """

        settings = self.settings
        now = time.time()
        async with self._lock:
            working, original_text, declared, error = self._load_working_lists()
            if error is not None:
                return error
            self._declared_models = declared
            current = working.get(task)
            if current is None:
                return f"未找到任务/模型类别 [{task}]，可用：{', '.join(sorted(working)) or '（空）'}"
            if model not in current:
                return f"模型 {model} 不在任务 [{task}] 的备选列表中，可用：{', '.join(current) or '（空）'}"
            if len(current) - 1 < settings.min_alternatives:
                return f"任务 [{task}] 移除后将只剩 {len(current) - 1} 个模型，低于保底，已取消"
            if self._store.state.freeze_for(task, model) is not None:
                return f"模型 {model} 已在任务 [{task}] 的冻结列表中"

            plan = settings.plan_for(task, model)
            duration = int(seconds) if int(seconds) > 0 else plan.duration(1)
            position = _position_of(current, model)
            style = ArrayStyle()
            with contextlib.suppress(ModelConfigError):
                style = self._host.array_style(original_text, task)
            working[task] = [name for name in current if name != model]
            try:
                result = self._host.apply(
                    {task: working[task]},
                    backup_dir=self._backup_dir,
                    backup_keep=max(0, settings.backup_keep),
                    dry_run=settings.dry_run,
                )
            except ModelConfigError as exc:
                self._store.bump("write_failed")
                return f"改写宿主模型配置失败：{exc}"

            self._note_write(result.written)

            entry = FreezeEntry(
                task=task,
                model=model,
                index=position.index,
                scope="task",
                frozen_at=now,
                until=now + duration,
                repeat=1,
                reason="手动冻结",
                applied=result.written,
                status_code=0,
                source="manual",
                after_model=position.after_model,
                before_model=position.before_model,
                array_style=style.to_dict(),
                context_models=list(current),
                left_models=[name for name in current if name != model],
            )
            self._store.state.add_freeze(entry)
            self._store.bump("frozen")
            self._persist()
            return (
                f"已手动冻结 [{task}] {model}，{format_duration(duration)}后自动还原（{plan.label}）"
                f"{'（演练模式未真正改写）' if not result.written else ''}"
            )

    async def reassert(self) -> List[str]:
        """把被外部改动放回列表的冻结模型重新移除。"""

        settings = self.settings
        if not settings.reassert_on_external_change:
            return []
        now = time.time()
        async with self._lock:
            working, _text, _declared, error = self._load_working_lists()
            if error is not None:
                return []
            updates: Dict[str, List[str]] = {}
            restored: List[str] = []
            for entry in self._store.state.freezes:
                if not entry.applied or entry.until <= now:
                    continue
                current = working.get(entry.task)
                if current is None or entry.model not in current:
                    continue
                if len(current) - 1 < settings.min_alternatives:
                    continue
                working[entry.task] = [name for name in current if name != entry.model]
                updates[entry.task] = working[entry.task]
                restored.append(f"[{entry.task}] {entry.model}")

            if not updates:
                return []
            try:
                result = self._host.apply(
                    updates,
                    backup_dir=self._backup_dir,
                    backup_keep=max(0, settings.backup_keep),
                    dry_run=settings.dry_run,
                )
            except ModelConfigError as exc:
                self._logger.warning(f"重新施加冻结失败：{exc}")
                return []
            self._note_write(result.written)
            message = f"检测到外部修改放回了冻结中的模型，已重新移除：{', '.join(restored)}"
            self._logger.info(message)
            return [message]

    def health_report(self, now: Optional[float] = None, ttl_seconds: float = 60.0) -> Dict[str, Any]:
        """配置体检：找出「任务列表引用了未声明模型」这类会让请求必然失败的问题。

        Args:
            now: 当前时间戳；留空取 ``time.time()``。
            ttl_seconds: 结果缓存时长（秒），避免每次调用都重新解析配置。

        Returns:
            dict[str, Any]: ``{"undeclared": {任务: [模型...]}, "checked_at": float}``。
        """

        current = time.time() if now is None else now
        if self._health_cache is not None and current - self._health_cached_at <= ttl_seconds:
            return self._health_cache

        undeclared: Dict[str, List[str]] = {}
        unused_rules: List[str] = []
        try:
            data = self._host.parse(self._host.read_text())
            self._declared_models = {str(name) for name in self._host.configured_model_names(data)}
            undeclared = self._host.undeclared_models(data)
            # 体检要按“配置的本来面貌”判断规则是否命中：被插件自己临时冻结、
            # 因而此刻不在列表里的模型，必须补回去，否则会误报规则未命中。
            health_lists = self._host.task_model_lists(data)
            for frozen in self._store.state.freezes:
                models = health_lists.setdefault(frozen.task, [])
                if frozen.model and frozen.model not in models:
                    models.append(frozen.model)
            unused_rules = unmatched_rules(self.settings.model_rules, health_lists)
        except ModelConfigError as exc:
            self._logger.debug(f"配置体检失败：{exc}")

        self._health_cache = {
            "undeclared": undeclared,
            "unused_rules": unused_rules,
            "checked_at": current,
        }
        self._health_cached_at = current
        return self._health_cache

    def health_lines(self) -> List[str]:
        """把配置体检结果整理成给用户看的文本行。"""

        report = self.health_report()
        undeclared = report.get("undeclared") or {}
        unused_rules = report.get("unused_rules") or []
        lines: List[str] = []

        if undeclared:
            lines.append(
                "⚠ 配置体检发现问题：以下任务引用了未在 [[models]] 中声明的模型，"
                "宿主调用时会直接报「未找到名为…的模型」并让该次请求失败："
            )
            for task, models in undeclared.items():
                lines.append(f"· [{task}] {'、'.join(models)}")
            lines.append("建议：在 model_config.toml 的 [[models]] 里补上声明，或从对应的 model_list 中删掉这些名字。")

        if unused_rules:
            lines.append("⚠ 以下「按模型冻结规则」没有匹配到任何现有模型，请检查通配符或任务名：")
            lines.extend(f"· {item}" for item in unused_rules)

        for problem in self.settings.rule_problems:
            lines.append(f"⚠ 冻结规则有问题：{problem}")

        return lines

    def describe_declared_state(self) -> str:
        """描述当前已声明模型数量（用于排查）。"""

        return f"已声明模型 {len(self._declared_models)} 个"

    # ------------------------------------------------------------------ 内部

    def _load_working_lists(self) -> Tuple[Dict[str, List[str]], str, Optional[str]]:
        """读取宿主配置中的任务模型列表。

        Returns:
            tuple[dict[str, list[str]], str, set[str], Optional[str]]:
            任务模型列表、原始文本、已声明模型名集合、错误信息。
        """

        try:
            text = self._host.read_text()
            data = self._host.parse(text)
        except ModelConfigError as exc:
            return {}, "", set(), f"读取宿主模型配置失败：{exc}"
        lists = self._host.task_model_lists(data)
        declared = {str(name) for name in self._host.configured_model_names(data)}
        return {task: list(models) for task, models in lists.items()}, text, declared, None

    @staticmethod
    def _dedupe_batch(detections: Sequence[DetectedRateLimit]) -> List[DetectedRateLimit]:
        """同一批里同一 (任务, 模型) 只保留一条。"""

        seen: set[Tuple[str, str]] = set()
        result: List[DetectedRateLimit] = []
        for detection in detections:
            key = (detection.task, detection.model)
            if key in seen:
                continue
            seen.add(key)
            result.append(detection)
        return result

    def _check_skippable(
        self,
        detection: DetectedRateLimit,
        settings: FailoverSettings,
        working: Dict[str, List[str]],
        declared: set[str],
        now: float,
    ) -> str:
        """返回跳过原因；空字符串表示需要处理。"""

        if not detection.task or not detection.model:
            return "事件缺少任务或模型信息"
        if settings.is_task_excluded(detection.task):
            return "该任务在排除列表中（如嵌入类任务）"
        if settings.is_model_excluded(detection.model):
            return "该模型在排除列表中"
        if detection.task not in working:
            return "宿主模型配置里没有这个任务"
        if detection.model not in working[detection.task]:
            return "该模型不在这个任务的备选列表里"

        if settings.skip_if_no_valid_alternative and declared:
            remaining = [name for name in working[detection.task] if name != detection.model]
            if remaining and not any(name in declared for name in remaining):
                return "该类别其余候选模型都没有在 [[models]] 中声明，冻结后请求会直接失败"

        existing = self._store.state.freeze_for(detection.task, detection.model)
        if existing is not None:
            if existing.until > now:
                if now - existing.frozen_at < BURST_DEDUPE_SECONDS:
                    return "同一轮限流的重复上报"
                return f"已在冻结中（剩余 {existing.describe_remaining(now)}）"
            return "已过期，等待还原后再处理"
        return ""

    def _target_tasks(
        self,
        detection: DetectedRateLimit,
        settings: FailoverSettings,
        working: Dict[str, List[str]],
    ) -> List[str]:
        """计算该模型需要被移除的任务列表。"""

        if not settings.global_scope:
            return [detection.task]
        return [
            task
            for task, models in working.items()
            if detection.model in models and not settings.is_task_excluded(task)
        ]

    def _next_repeat(self, task: str, model: str, now: float, settings: FailoverSettings) -> int:
        """计算重复次数（用于冻结时长升级）。

        参照点是「最近一次尝试该模型的时间」= max(上次冻结开始, 上次还原时间)。
        只看冻结开始会让长时间冻结（可能长达 30 分钟）把判定窗口熬过去，
        于是每次都是「180 秒 → 到期还原 → 立刻又 429 → 又重新从 180 秒开始」，
        永远升不上去。改成看还原时间后，只要模型一放回去就立刻再被限流，
        冻结时长就会持续升到上限；等它真的恢复（窗口内没有再 429）才回落。
        """

        plan = settings.plan_for(task, model)
        if not plan.escalate:
            return 1
        window = plan.window_seconds
        if window <= 0:
            return 1

        history = self._load_history()
        record = history.get(f"{task}\u0000{model}")
        if not isinstance(record, list) or len(record) < 2:
            return 1
        with contextlib.suppress(TypeError, ValueError):
            frozen_at = float(record[0])
            last_repeat = int(record[1])
            restored_at = float(record[2]) if len(record) > 2 and record[2] else 0.0
            reference = max(frozen_at, restored_at)
            if now - reference <= window:
                if last_repeat + 1 > 10:
                    return 10
                return last_repeat + 1
        return 1

    def _remember_history(self, task: str, model: str, now: float, repeat: int) -> None:
        """记录最近一次冻结，用于时长升级。"""

        history = self._load_history()
        history[f"{task}\u0000{model}"] = [now, repeat]
        self._store.note("history", history)

    def _remember_restore(self, entry: FreezeEntry, now: float) -> None:
        """记录最近一次还原时间，用于判定「放回去立刻又被限流」。"""

        history = self._load_history()
        history[entry.key] = [entry.frozen_at, entry.repeat, now]
        self._store.note("history", history)

    def _load_history(self) -> Dict[str, Any]:
        """读取冻结历史表。"""

        history = self._store.state.stats.get("history")
        return history if isinstance(history, dict) else {}

    async def _restore(self, entries: Sequence[FreezeEntry], reason: str) -> List[str]:
        """执行还原。"""

        settings = self.settings
        now = time.time()
        messages: List[str] = []
        async with self._lock:
            working, _text, _declared, error = self._load_working_lists()
            if error is not None:
                self._logger.warning(f"还原冻结时{error}")
                for entry in entries:
                    self._restore_retry_at[entry.key] = now + RESTORE_RETRY_SECONDS
                return [error]

            updates: Dict[str, List[str]] = {}
            styles: Dict[str, ArrayStyle] = {}
            restorable: List[FreezeEntry] = []
            skipped_manual: List[FreezeEntry] = []
            manual_edit_removals: List[str] = []
            manual_edit_additions: List[str] = []
            grouped = self._group_entries(entries)
            # 本批还原之外、仍然生效的冻结条目（它们移除的模型同样是“我们干的”）
            pending_removed: Dict[str, set] = {}
            batch_keys = {item.key for item in entries}
            for item in self._store.state.freezes:
                if item.applied and item.key not in batch_keys:
                    pending_removed.setdefault(item.task, set()).add(item.model)
            for entry in sorted(entries, key=lambda item: (item.task, item.frozen_at, item.index)):
                if not entry.applied:
                    messages.append(f"[{entry.task}] {entry.model} 为演练模式记录，无需还原")
                    continue
                current = working.get(entry.task)
                if current is None:
                    messages.append(f"[{entry.task}] 已不存在于模型配置，跳过 {entry.model} 的还原")
                    continue
                # 判断这份列表是不是被外部改过：把我们自己移掉的模型（本批与仍在冻结中的）
                # 从“离开时的样子”里扣掉，再看是否有模型凭空消失。只增不减不算冲突
                # （新增的模型会被保留，我们按锚点把模型插回去即可）。
                removed_by_us = set(pending_removed.get(entry.task, set())) | {
                    item.model for item in entries if item.task == entry.task
                }
                expected = [name for name in entry.left_models if name not in removed_by_us]
                missing_names = [name for name in expected if name not in current]
                edited_externally = bool(entry.left_models) and list(current) != list(entry.left_models)
                removed_externally = bool(missing_names)
                if edited_externally:
                    target = manual_edit_removals if removed_externally else manual_edit_additions
                    if entry.task not in target:
                        target.append(entry.task)
                if entry.model in current:
                    messages.append(f"[{entry.task}] {entry.model} 已在列表中，无需还原")
                    continue
                if removed_externally and settings.respect_manual_edit:
                    # 冻结期间有人从列表里删掉了原本存在的模型（且我们冻结的这个也没回来）：
                    # 说明对方（很可能是用户）在重做这份列表，此时把模型插回去等于用插件的
                    # 临时改动覆盖人的编辑，因此跳过还原，只做提示。
                    skipped_manual.append(entry)
                    messages.append(
                        f"[{entry.task}] {entry.model} 未还原：冻结期间列表被外部修改且已不含该模型，"
                        "按「尊重手工改动」跳过（如需它回到列表，请在配置里重新加入，"
                        "或关闭「尊重手工改动」）"
                    )
                    continue
                style = ArrayStyle.from_dict(entry.array_style)
                if style is not None:
                    styles.setdefault(entry.task, style)
                position = self._derive_position(entry, grouped.get(entry.task, []))
                current.insert(self._restore_index(current, position), entry.model)
                working[entry.task] = list(current)
                updates[entry.task] = list(current)
                restorable.append(entry)

            if updates:
                try:
                    result = self._host.apply(
                        updates,
                        styles=styles,
                        backup_dir=self._backup_dir,
                        backup_keep=max(0, settings.backup_keep),
                        dry_run=False,
                    )
                except ModelConfigError as exc:
                    self._store.bump("write_failed")
                    self._logger.error(f"{reason}时还原宿主模型配置失败，{int(RESTORE_RETRY_SECONDS)} 秒后重试：{exc}")
                    for entry in restorable:
                        self._restore_retry_at[entry.key] = now + RESTORE_RETRY_SECONDS
                    return [f"还原失败：{exc}"]

                self._note_write(result.written)

                for entry in restorable:
                    self._restore_retry_at.pop(entry.key, None)
                    self._remember_restore(entry, now)
                    messages.append(f"[{entry.task}] {entry.model} 已还原到原位置")
                    self._store.bump("restored")

            for entry in entries:
                self._store.state.remove_freeze(entry.task, entry.model)
            self._persist()

            if manual_edit_removals:
                self._logger.warning(
                    f"冻结期间任务 {('、'.join(manual_edit_removals))} 的 model_list 被手工改过"
                    "（有模型被移除）：还原以你的改动为准，被冻结的模型若仍需要请手动加回"
                )
            if manual_edit_additions:
                self._logger.info(
                    f"冻结期间任务 {('、'.join(manual_edit_additions))} 的 model_list 有新增模型，"
                    "已保留这些新增并还原被冻结的模型"
                )
            if skipped_manual:
                self._logger.info(
                    "以下冻结因「列表被手工修改且已不含该模型」而跳过还原："
                    + "、".join(f"[{item.task}] {item.model}" for item in skipped_manual)
                )

            if messages:
                self._store.note("last_event", {"summary": f"{reason}：" + "；".join(messages), "at": now})
                self._logger.info(f"{reason}：" + "；".join(messages))

        if settings.notify_enabled and settings.notify_on_unfreeze and updates:
            await self._notify(f"{reason}：\n" + "\n".join(messages))
        return messages

    @staticmethod
    def _group_entries(entries: Sequence[FreezeEntry]) -> Dict[str, List[FreezeEntry]]:
        """按任务分组冻结条目。"""

        grouped: Dict[str, List[FreezeEntry]] = {}
        for entry in entries:
            grouped.setdefault(entry.task, []).append(entry)
        return grouped

    @staticmethod
    def _derive_position(entry: FreezeEntry, siblings: Sequence[FreezeEntry]) -> _ModelPosition:
        """推导还原时的锚点与下标。

        同一个模型可能是在列表已经被别人缩短之后才被冻结的（此时它的直接前驱
        已经不在列表里），因此这里优先从**最早的**、包含该模型的上下文列表里
        读取它当时的真实前驱/后继；找不到上下文时退回记录下来的锚点与下标。
        """

        candidates = [item for item in siblings if entry.model in item.context_models]
        if candidates:
            earliest = min(candidates, key=lambda item: item.frozen_at)
            names = [str(name) for name in earliest.context_models]
            index = names.index(entry.model)
            return _ModelPosition(
                index=index,
                after_model=names[index - 1] if index > 0 else "",
                before_model=names[index + 1] if index + 1 < len(names) else "",
            )
        return _ModelPosition(
            index=int(entry.index),
            after_model=entry.after_model,
            before_model=entry.before_model,
        )

    @staticmethod
    def _restore_index(current: Sequence[str], position: _ModelPosition) -> int:
        """按锚点计算插入下标。

        优先插到「原本的后继模型」之前，其次插到「原本的前驱模型」之后，
        两者都不在列表里时退回记录的下标（并夹在合法范围内）。
        """

        names = [str(item) for item in current]
        if position.before_model and position.before_model in names:
            return names.index(position.before_model)
        if position.after_model and position.after_model in names:
            return names.index(position.after_model) + 1
        return max(0, min(int(position.index), len(names)))

    def touch_config(self) -> bool:
        """就地把当前配置原样重写一遍，强制产生一次「文件已修改」事件。

        用途：宿主刚启动时，配置监听或插件订阅可能还没就绪，插件写进去的改动
        虽然最终会被宿主在启动读取配置时采纳，但插件收不到那次热重载回执。
        这里主动再写一次（内容不变、inode 不变），让已经就绪的监听重新触发一次
        热重载与广播，避免把「没收到回执」误判成「监听失效」。
        """

        try:
            text = self._host.read_text()
            self._host.write_in_place(text)
        except ModelConfigError as exc:
            self._logger.warning(f"重写宿主模型配置以触发热重载失败：{exc}")
            return False
        self._note_write(True)
        self._logger.info("已就地重写宿主模型配置（内容不变），用于触发一次热重载确认")
        return True

    def _persist(self) -> None:
        """写回状态文件。"""

        self._store.save()

    def _note_write(self, written: bool) -> None:
        """标记一次由本插件发起的宿主配置写入（避免把自己的写入当成外部改动）。"""

        if not written or self._on_write is None:
            return
        with contextlib.suppress(Exception):
            self._on_write()

    async def _notify(self, text: str) -> None:
        """发送通知（失败不影响主流程）。"""

        if self._notifier is None or not text.strip():
            return
        try:
            result = self._notifier(text)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # noqa: BLE001 - 通知失败不能影响冻结逻辑
            self._logger.warning(f"发送冻结通知失败：{exc}")

    @staticmethod
    def _describe_alternatives(models: Iterable[str], declared: Optional[set[str]] = None) -> str:
        """描述剩余备选模型；未在 ``[[models]]`` 中声明的会额外标注。"""

        names = [str(name) for name in models]
        if not names:
            return "（已无其他备选，请尽快补充模型配置）"
        rendered = [
            f"{name}（⚠未声明，调用会失败）" if declared and name not in declared else name
            for name in names
        ]
        return "、".join(rendered)

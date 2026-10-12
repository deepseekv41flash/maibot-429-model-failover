"""离线自测：不依赖宿主与 maibot_sdk，只用标准库验证核心逻辑。

运行方式（容器内，推荐）::

    docker exec -w /MaiMBot/plugins/maibot-429-model-failover maim-bot-core \
        .venv/bin/python tests/test_offline.py

本机 Python ≥3.11 亦可直接运行。覆盖点：

* 宿主机型配置的“外科式改写 → 还原”字节级 round-trip；
* 解析后逐节比对的防误改校验；
* 失败快照与应用日志的限流事件识别；
* 冻结管理器的完整流程（冻结 → 列表变化 → 到期还原 → 保底/排除/全局范围）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import sys
import tempfile
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from failover.detector import DetectedRateLimit, FailureDetector  # noqa: E402
from failover.freeze import FreezeManager  # noqa: E402
from failover.host_config import HostModelConfig, ModelConfigError  # noqa: E402
from failover.rules import (  # noqa: E402
    build_rules,
    format_duration,
    parse_duration,
    resolve_plan,
    unmatched_rules,
)
from failover.settings import FailoverSettings, build_settings  # noqa: E402
from failover.state import StateStore  # noqa: E402

SAMPLE_CONFIG = '''[inner]
version = "1.17.15"

[[models]]
model_identifier = "glm-4.6v-flash"
name = "glm-4.6v-flash"
api_provider = "ZhipuAI"
visual = true

[models.extra_params]

[[models]]
model_identifier = "embedding-3"
name = "embedding-3"
api_provider = "ZhipuAI"

[models.extra_params]

[[models]]
model_identifier = "deepseek-v4-flash"
name = "deepseek-v4-flash"
api_provider = "DeepSeek"
visual = false

[models.extra_params]

[model_task_config.replyer] # 回复模型
model_list = [
    "glm-4.6v-flash",
    "deepseek-v4-flash",
] # 使用的模型列表, 每个元素对应上面的模型名称(name)
max_tokens = 4096 # 任务最大输出token数
temperature = 1 # 模型温度
selection_strategy = "sequential" # 模型选择策略

[model_task_config.utils]
model_list = ["deepseek-v4-flash"] # 单行数组写法
max_tokens = 8000
selection_strategy = "balance"

[model_task_config.fast_model]
model_list = [
    "deepseek-v4-flash",
    "glm-4.6v-flash",
] # 快速模型
selection_strategy = "balance"

[model_task_config.embedding]
model_list = ["embedding-3"] # 嵌入
selection_strategy = "sequential"

[[api_providers]]
name = "ZhipuAI"
base_url = "https://example.invalid/api"
api_key = "sk-test"

[api_providers.default_headers]
'''

_LOGGER_LINES: List[str] = []


class ListLogger:
    """把日志收进列表的简易 logger。"""

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """记录 debug 日志。"""

        del args, kwargs
        _LOGGER_LINES.append(f"debug: {message}")

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """记录 info 日志。"""

        del args, kwargs
        _LOGGER_LINES.append(f"info: {message}")

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """记录 warning 日志。"""

        del args, kwargs
        _LOGGER_LINES.append(f"warning: {message}")

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """记录 error 日志。"""

        del args, kwargs
        _LOGGER_LINES.append(f"error: {message}")


def _write(path: Path, text: str) -> None:
    """写入文本文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _settings(**overrides: Any) -> FailoverSettings:
    """构造测试用配置快照。"""

    base: Dict[str, Any] = {
        "freeze_seconds": 2,
        "escalate": False,
        "exclude_tasks": ("embedding", "image_embedding"),
        "min_alternatives": 1,
        "detect_rate_limit_by_type": True,
    }
    base.update(overrides)
    return FailoverSettings(**base)


def _make_workspace(tmp: Path) -> tuple[Path, HostModelConfig, StateStore, ListLogger]:
    """准备一份临时宿主配置与状态仓库。"""

    config_path = tmp / "config" / "model_config.toml"
    _write(config_path, SAMPLE_CONFIG)
    logger = ListLogger()
    return config_path, HostModelConfig(config_path, logger), StateStore(tmp / "state.json", logger), logger


def _make_manager(
    config_path: Path,
    host: HostModelConfig,
    store: StateStore,
    logger: ListLogger,
    settings: FailoverSettings,
) -> FreezeManager:
    """构造冻结管理器。"""

    notices: List[str] = []
    manager = FreezeManager(
        host_config=host,
        store=store,
        logger=logger,
        settings_provider=lambda: settings,
        backup_dir=config_path.parent / "backups",
        notifier=notices.append,
    )
    manager.notices = notices  # type: ignore[attr-defined]
    return manager


# --------------------------------------------------------------------- 编辑器


def test_editor_roundtrip_multiline() -> None:
    """多行数组：移除 → 还原后必须与原文件逐字节一致。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        config_path, host, _store, logger = _make_workspace(tmp)
        original = config_path.read_text(encoding="utf-8")

        result = host.apply({"replyer": ["deepseek-v4-flash"]}, dry_run=True)
        assert result.changed and not result.written
        assert config_path.read_text(encoding="utf-8") == original, "演练模式不应写盘"

        result = host.apply({"replyer": ["deepseek-v4-flash"]}, backup_dir=tmp / "backups")
        assert result.written, "应真实写入"
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        assert data["model_task_config"]["replyer"]["model_list"] == ["deepseek-v4-flash"]
        assert (tmp / "backups").glob("model_config_*.toml"), "应生成备份"

        host.apply({"replyer": ["glm-4.6v-flash", "deepseek-v4-flash"]})
        assert config_path.read_text(encoding="utf-8") == original, "还原后文件必须与原始内容一致"
        assert _LOGGER_LINES is not None and logger is not None


def test_editor_roundtrip_single_line() -> None:
    """单行数组：保持单行风格并正确还原。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        config_path, host, _store, _logger = _make_workspace(tmp)
        original = config_path.read_text(encoding="utf-8")

        host.apply({"utils": []})
        assert "model_list = []" in config_path.read_text(encoding="utf-8")

        host.apply({"utils": ["deepseek-v4-flash"]})
        assert "model_list = [\"deepseek-v4-flash\"]" in config_path.read_text(encoding="utf-8")
        assert config_path.read_text(encoding="utf-8") == original


def test_editor_roundtrip_every_task() -> None:
    """对每个任务逐一执行一次移除/还原，文件都应回到原样。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        config_path, host, _store, _logger = _make_workspace(tmp)
        original = config_path.read_text(encoding="utf-8")
        data = tomllib.loads(original)
        for task, models in host.task_model_lists(data).items():
            if len(models) < 2:
                continue
            host.apply({task: models[1:]})
            host.apply({task: list(models)})
            assert config_path.read_text(encoding="utf-8") == original, f"任务 {task} 还原不一致"


def test_editor_keeps_inode() -> None:
    """写入必须保持 inode 不变。

    宿主用 ``watchfiles`` 监听 ``model_config.toml``，inotify 监听挂在 inode 上；
    一旦用 ``os.replace`` 换掉 inode，监听会静默失效（宿主既不再热重载，也不会
    自动恢复），所以这里锁死「写入不换 inode」这一行为。
    """

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        config_path, host, _store, _logger = _make_workspace(tmp)
        inode_before = config_path.stat().st_ino
        original = config_path.read_text(encoding="utf-8")

        host.apply({"replyer": ["deepseek-v4-flash"]})
        assert config_path.stat().st_ino == inode_before, "冻结写入不得更换 inode（会破坏宿主文件监听）"

        host.apply({"replyer": ["glm-4.6v-flash", "deepseek-v4-flash"]})
        assert config_path.stat().st_ino == inode_before, "还原写入同样不得更换 inode"
        assert config_path.read_text(encoding="utf-8") == original


def test_editor_rejects_unexpected_change() -> None:
    """除目标数组外的任何差异都必须被拒绝。"""

    text = SAMPLE_CONFIG
    tampered = text.replace('temperature = 1 # 模型温度', 'temperature = 0.5 # 模型温度')
    with tempfile.TemporaryDirectory() as raw_tmp:
        config_path, host, _store, _logger = _make_workspace(Path(raw_tmp))
        try:
            host.validate_change(text, tampered, {"replyer": ["deepseek-v4-flash"]})
        except ModelConfigError:
            return
        raise AssertionError("被篡改的配置本应校验失败")


def test_editor_rejects_broken_toml() -> None:
    """语法损坏的文本必须被拒绝。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        _config_path, host, _store, _logger = _make_workspace(Path(raw_tmp))
        try:
            host.parse("[inner]\nversion = ")
        except ModelConfigError:
            return
        raise AssertionError("非法 TOML 本应抛错")


def test_real_config_roundtrip() -> None:
    """若本机存在真实宿主配置，则对它做只读的拷贝 round-trip。"""

    candidates = [
        Path("/MaiMBot/config/model_config.toml"),
        Path("/root/data/MaiMBot/config/model_config.toml"),
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        print("    （跳过：本机没有真实宿主配置）")
        return

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        config_path = tmp / "config" / "model_config.toml"
        _write(config_path, source.read_text(encoding="utf-8"))
        original = config_path.read_text(encoding="utf-8")
        logger = ListLogger()
        host = HostModelConfig(config_path, logger)
        data = host.parse(original)
        tasks = host.task_model_lists(data)
        assert tasks, "真实配置里应至少有一个任务表"

        checked = 0
        for task, models in tasks.items():
            if len(models) < 2:
                continue
            stripped = models[1:]
            host.apply({task: stripped})
            now = host.task_model_lists(host.parse(config_path.read_text(encoding="utf-8")))
            assert now[task] == stripped, f"任务 {task} 写入结果不符"
            host.apply({task: list(models)})
            assert config_path.read_text(encoding="utf-8") == original, f"任务 {task} 还原不一致"
            checked += 1
        print(f"    （真实配置 round-trip 通过，覆盖 {checked} 个多模型任务）")


# --------------------------------------------------------------------- 检测


def test_snapshot_detection() -> None:
    """失败快照里的 429 尝试应被识别。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        error_dir = tmp / "logs" / "maisaka_prompt" / "llm_error" / "system"
        snapshot = {
            "request": {"task_name": "utils"},
            "metadata": {"model_name": "inclusionai/ling-3.1-flash:free", "status": "switching_model"},
            "generation_attempts": [
                {
                    "attempt_id": "abc:1",
                    "model": "inclusionai/ling-3.1-flash:free",
                    "error": {"status_code": 429, "type": "RateLimitError", "message": "Error code: 429 - rate limit"},
                }
            ],
        }
        _write(error_dir / "20260101_000000_deadbeef.json", json.dumps(snapshot))

        detector = FailureDetector(logger=ListLogger(), error_dir=error_dir, log_dir=None, status_codes=(429,))
        found = detector.scan()
        assert len(found) == 1, f"应识别 1 条，实际 {len(found)}"
        assert found[0].task == "utils"
        assert found[0].model == "inclusionai/ling-3.1-flash:free"
        assert found[0].status_code == 429

        # 检测器本身无状态，重复扫描会再次产出同一 event_id；
        # 真正的去重由插件的 StateStore（跨重启持久化）完成。
        again = detector.scan()
        assert len(again) == 1 and again[0].event_id == found[0].event_id
        store = StateStore(tmp / "state.json", ListLogger())
        store.load()
        assert store.mark_processed(found[0].event_id) is True
        assert store.mark_processed(again[0].event_id) is False, "同一事件不应被处理两次"

        # 新请求产生新的 attempt_id → 可再次识别
        snapshot["generation_attempts"][0]["attempt_id"] = "abc:2"
        _write(error_dir / "20260101_000100_deadbeef.json", json.dumps(snapshot))
        found = detector.scan()
        assert len(found) == 2, "新旧两份快照都会被扫到"
        assert any(item.event_id.endswith("abc:2") for item in found)
        assert store.mark_processed("snapshot:abc:2") is True


def test_log_detection() -> None:
    """真实格式的宿主日志行应被识别，且支持增量读取。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        log_dir = tmp / "logs"
        log_path = log_dir / "app_20260101_000000.log.jsonl"
        retry_line = {
            "logger_name": "llm_models",
            "event": (
                "任务 'utils' 的模型 'inclusionai/ling-3.1-flash:free' 遇到可重试的HTTP错误: 请求过于频繁，请稍后再试\n"
                "  底层异常: RateLimitError | Error code: 429 - {'error': {'type': 'rate_limit_error'}}。剩余重试次数: 2"
            ),
            "level": "warning",
        }
        final_line = {
            "logger_name": "llm_models",
            "event": "任务 'planner' 的模型 'deepseek-v4-flash' 在遇到 429 错误并用尽重试次数后仍然失败。",
            "level": "error",
        }
        _write(log_path, json.dumps(retry_line, ensure_ascii=False) + "\n" + json.dumps(final_line, ensure_ascii=False) + "\n")

        detector = FailureDetector(logger=ListLogger(), error_dir=None, log_dir=log_dir, status_codes=(429,))
        found = detector.scan()
        assert len(found) == 2, f"应识别 2 条，实际 {len(found)}: {[item.describe() for item in found]}"
        assert {item.task for item in found} == {"utils", "planner"}
        assert detector.scan() == [], "增量读取不应重复消费同一行"

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final_line, ensure_ascii=False) + "\n")
        found = detector.scan()
        assert len(found) == 1, "追加的行应被读到"


def test_log_detection_ignores_other_errors() -> None:
    """非限流错误不应触发。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        log_dir = tmp / "logs"
        line = {
            "logger_name": "llm_models",
            "event": "任务 'utils' 的模型 'glm-4.6v-flash' 遇到不可重试的HTTP错误: 请求参数错误。",
            "level": "warning",
        }
        _write(log_dir / "app_20260101_000000.log.jsonl", json.dumps(line, ensure_ascii=False) + "\n")
        detector = FailureDetector(logger=ListLogger(), error_dir=None, log_dir=log_dir, status_codes=(429,))
        assert detector.scan() == []


# --------------------------------------------------------------------- 冻结


def _run(coro: Any) -> Any:
    """运行协程。"""

    return asyncio.run(coro)


def test_manager_freeze_and_restore() -> None:
    """冻结 → 列表变化 → 到期 → 还原完整流程。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            original = config_path.read_text(encoding="utf-8")
            settings = _settings(freeze_seconds=2, escalate=False)
            manager = _make_manager(config_path, host, store, logger, settings)

            detection = DetectedRateLimit(
                task="replyer",
                model="glm-4.6v-flash",
                status_code=429,
                source="snapshot",
                event_id="snapshot:test:1",
                detected_at=time.time(),
            )
            messages = await manager.process([detection])
            assert messages and "已冻结" in messages[0], messages

            lists = host.task_model_lists(host.parse(config_path.read_text(encoding="utf-8")))
            assert lists["replyer"] == ["deepseek-v4-flash"], lists["replyer"]
            assert store.state.freeze_for("replyer", "glm-4.6v-flash") is not None

            # 冻结期内重复上报不应重复计数
            repeat_messages = await manager.process([detection])
            assert all("跳过" in message for message in repeat_messages), repeat_messages
            assert store.state.stats.get("frozen") == 1

            await asyncio.sleep(2.1)
            restored = await manager.expire_due()
            assert restored, "到期应触发还原"
            assert config_path.read_text(encoding="utf-8") == original, "还原后文件必须与原始内容一致"
            assert store.state.freezes == []
            assert int(store.state.stats.get("restored", 0)) == 1

    _run(scenario())


def test_manager_keeps_last_model() -> None:
    """只剩一个模型的任务不会被冻结，避免请求直接失败。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            before = config_path.read_text(encoding="utf-8")
            manager = _make_manager(config_path, host, store, logger, _settings())
            messages = await manager.process(
                [
                    DetectedRateLimit(
                        task="utils",
                        model="deepseek-v4-flash",
                        status_code=429,
                        source="log",
                        event_id="log:test:1",
                        detected_at=time.time(),
                    )
                ]
            )
            assert messages and "跳过" in messages[0], messages
            assert config_path.read_text(encoding="utf-8") == before
            assert store.state.freezes == []

    _run(scenario())


def test_manager_excludes_embedding_task() -> None:
    """嵌入类任务默认排除。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            before = config_path.read_text(encoding="utf-8")
            manager = _make_manager(config_path, host, store, logger, _settings(exclude_tasks=("embedding",)))
            messages = await manager.process(
                [
                    DetectedRateLimit(
                        task="embedding",
                        model="embedding-3",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:e:1",
                        detected_at=time.time(),
                    )
                ]
            )
            assert messages and "跳过" in messages[0]
            assert config_path.read_text(encoding="utf-8") == before

    _run(scenario())


def test_manager_restore_order_after_batch_freeze() -> None:
    """一次批量冻结多个模型后，还原必须保持原有顺序。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path = tmp / "config" / "model_config.toml"
            _write(config_path, SAMPLE_CONFIG.replace(
                '[model_task_config.utils]\nmodel_list = ["deepseek-v4-flash"] # 单行数组写法',
                '[model_task_config.utils]\nmodel_list = [\n    "deepseek-v4-flash",\n    "glm-4.6v-flash",\n] # 多模型',
            ))
            logger = ListLogger()
            host = HostModelConfig(config_path, logger)
            store = StateStore(tmp / "state.json", logger)
            store.load()
            original = config_path.read_text(encoding="utf-8")
            settings = _settings(freeze_seconds=1, freeze_scope="model", min_alternatives=0)
            manager = _make_manager(config_path, host, store, logger, settings)

            now = time.time()
            detections = [
                DetectedRateLimit(
                    task="utils",
                    model=model,
                    status_code=429,
                    source="snapshot",
                    event_id=f"snapshot:order:{index}",
                    detected_at=now,
                )
                for index, model in enumerate(["deepseek-v4-flash", "glm-4.6v-flash"])
            ]
            await manager.process(detections)

            lists = host.task_model_lists(host.parse(config_path.read_text(encoding="utf-8")))
            assert lists["utils"] == [], f"两个模型都应被移除，实际 {lists['utils']}"
            indexes = sorted(entry.index for entry in store.state.freezes if entry.task == "utils")
            assert indexes == [0, 1], f"索引必须基于原始列表计算，实际 {indexes}"

            await asyncio.sleep(1.1)
            await manager.expire_due()
            assert config_path.read_text(encoding="utf-8") == original, "还原顺序必须与原始文件一致"

    _run(scenario())


def test_manager_restore_order_after_staggered_freeze() -> None:
    """分批冻结（先后两次）后，还原同样必须保持原有顺序。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            original = config_path.read_text(encoding="utf-8")
            settings = _settings(freeze_seconds=1, min_alternatives=0)
            manager = _make_manager(config_path, host, store, logger, settings)

            # replyer 原列表：glm-4.6v-flash, deepseek-v4-flash
            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer",
                        model="glm-4.6v-flash",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:stagger:1",
                        detected_at=time.time(),
                    )
                ]
            )
            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer",
                        model="deepseek-v4-flash",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:stagger:2",
                        detected_at=time.time(),
                    )
                ]
            )
            lists = host.task_model_lists(host.parse(config_path.read_text(encoding="utf-8")))
            assert lists["replyer"] == [], f"两个模型都应被移除，实际 {lists['replyer']}"

            await asyncio.sleep(1.1)
            await manager.expire_due()
            assert config_path.read_text(encoding="utf-8") == original

    _run(scenario())


def test_manager_global_scope() -> None:
    """freeze_scope=model 时应从所有出现该模型的任务里移除。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            original = config_path.read_text(encoding="utf-8")
            settings = _settings(freeze_scope="model", freeze_seconds=2)
            manager = _make_manager(config_path, host, store, logger, settings)

            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer",
                        model="deepseek-v4-flash",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:g:1",
                        detected_at=time.time(),
                    )
                ]
            )
            lists = host.task_model_lists(host.parse(config_path.read_text(encoding="utf-8")))
            assert "deepseek-v4-flash" not in lists["replyer"]
            assert "deepseek-v4-flash" not in lists["fast_model"], "全局范围应同时移除其它任务里的同一模型"
            assert lists["utils"] == ["deepseek-v4-flash"], "只剩一个模型的任务不受影响"

            await asyncio.sleep(2.1)
            await manager.expire_due()
            assert config_path.read_text(encoding="utf-8") == original, "全局冻结同样要完整还原"

    _run(scenario())


def test_manager_escalation_and_dry_run() -> None:
    """演练模式不写盘；开启翻倍后冻结时长递增。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            before = config_path.read_text(encoding="utf-8")
            settings = _settings(dry_run=True, freeze_seconds=10, escalate=True, max_freeze_seconds=25)
            manager = _make_manager(config_path, host, store, logger, settings)

            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer",
                        model="glm-4.6v-flash",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:d:1",
                        detected_at=time.time(),
                    )
                ]
            )
            assert config_path.read_text(encoding="utf-8") == before, "演练模式不得写盘"
            entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
            assert entry is not None and entry.applied is False
            assert abs(entry.until - entry.frozen_at - 10) < 0.5, entry

            assert settings.freeze_duration(1) == 10
            assert settings.freeze_duration(2) == 20
            assert settings.freeze_duration(3) == 25, "不得超过上限"

            entry.until = time.time() - 1
            await manager.expire_due()
            assert store.state.freezes == [], "演练模式的记录到期后应被清理"

    _run(scenario())


def test_manager_dry_run_entry_does_not_block_real_freeze() -> None:
    """关闭演练模式后，演练记录不应继续占用冻结名额。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            original = config_path.read_text(encoding="utf-8")

            dry_manager = _make_manager(
                config_path, host, store, logger, _settings(dry_run=True, freeze_seconds=600)
            )
            detection = DetectedRateLimit(
                task="replyer",
                model="glm-4.6v-flash",
                status_code=429,
                source="snapshot",
                event_id="snapshot:dry:1",
                detected_at=time.time(),
            )
            await dry_manager.process([detection])
            assert store.state.freeze_for("replyer", "glm-4.6v-flash") is not None
            assert config_path.read_text(encoding="utf-8") == original, "演练模式不写盘"

            real_manager = _make_manager(
                config_path, host, store, logger, _settings(dry_run=False, freeze_seconds=600)
            )
            await real_manager.process(
                [
                    DetectedRateLimit(
                        task="replyer",
                        model="glm-4.6v-flash",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:dry:2",
                        detected_at=time.time(),
                    )
                ]
            )
            lists = host.task_model_lists(host.parse(config_path.read_text(encoding="utf-8")))
            assert lists["replyer"] == ["deepseek-v4-flash"], "真实冻结应当立刻生效"
            entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
            assert entry is not None and entry.applied is True

    _run(scenario())


def test_log_detection_drops_stale_events() -> None:
    """重启后重新定位日志时，几百行以前的历史 429 不应被当成刚发生的事。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        log_dir = tmp / "logs"
        old_day = (datetime.now() - timedelta(hours=5)).strftime("%m-%d %H:%M:%S")
        line_old = {
            "logger_name": "llm_models",
            "event": "任务 'utils' 的模型 'glm-4.6v-flash' 遇到可重试的HTTP错误: Error code: 429。剩余重试次数: 2",
            "timestamp": old_day,
            "level": "warning",
        }
        line_new = {
            "logger_name": "llm_models",
            "event": "任务 'utils' 的模型 'deepseek-v4-flash' 遇到可重试的HTTP错误: Error code: 429。剩余重试次数: 2",
            "timestamp": datetime.now().strftime("%m-%d %H:%M:%S"),
            "level": "warning",
        }
        line_plugin_iso = {
            "logger": "plugin.example.llm-stream",
            "event": "任务 'utils' 的模型 'glm-4.6v-flash' 遇到可重试的HTTP错误: Error code: 429。剩余重试次数: 1",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "level": "warning",
        }
        _write(
            log_dir / "app_20260101_000000.log.jsonl",
            "\n".join(json.dumps(item, ensure_ascii=False) for item in (line_old, line_new, line_plugin_iso)) + "\n",
        )

        detector = FailureDetector(
            logger=ListLogger(),
            error_dir=None,
            log_dir=log_dir,
            status_codes=(429,),
            max_log_event_age_seconds=300,
        )
        found = detector.scan()
        models = sorted(item.model for item in found)
        assert models == ["deepseek-v4-flash", "glm-4.6v-flash"], f"过旧事件应被丢弃，实际 {models}"
        assert detector._stale_log_skipped == 1

        # 关闭年龄限制后全部接受
        detector2 = FailureDetector(
            logger=ListLogger(),
            error_dir=None,
            log_dir=log_dir,
            status_codes=(429,),
            max_log_event_age_seconds=0,
        )
        assert len(detector2.scan()) == 3


def test_health_report_finds_undeclared_models() -> None:
    """任务列表里引用了未声明的模型时必须能查出来，并据此跳过冻结。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path = tmp / "config" / "model_config.toml"
            # replyer 只剩一个已声明模型 + 一个未声明模型
            _write(
                config_path,
                SAMPLE_CONFIG.replace(
                    '    "deepseek-v4-flash",\n] # 使用的模型列表',
                    '    "ghost-model",\n] # 使用的模型列表',
                ),
            )
            logger = ListLogger()
            host = HostModelConfig(config_path, logger)
            store = StateStore(tmp / "state.json", logger)
            store.load()
            manager = _make_manager(config_path, host, store, logger, _settings(freeze_seconds=1))

            report = manager.health_report()
            assert report["undeclared"] == {"replyer": ["ghost-model"]}, report
            assert manager.health_lines(), "应产出可读的体检提示"

            before = config_path.read_text(encoding="utf-8")
            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer",
                        model="glm-4.6v-flash",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:health:1",
                        detected_at=time.time(),
                    )
                ]
            )
            assert store.state.freezes == [], "剩下的备选模型都未声明时应跳过冻结"
            assert config_path.read_text(encoding="utf-8") == before

            # 关闭该护栏后照常冻结
            relaxed = _make_manager(
                config_path,
                host,
                store,
                logger,
                _settings(freeze_seconds=1, skip_if_no_valid_alternative=False),
            )
            await relaxed.process(
                [
                    DetectedRateLimit(
                        task="replyer",
                        model="glm-4.6v-flash",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:health:2",
                        detected_at=time.time(),
                    )
                ]
            )
            assert store.state.freeze_for("replyer", "glm-4.6v-flash") is not None
            lists = host.task_model_lists(host.parse(config_path.read_text(encoding="utf-8")))
            assert lists["replyer"] == ["ghost-model"]

    _run(scenario())


def test_escalation_counts_restore_time() -> None:
    """放回去立刻又被限流时，冻结时长应继续升级，而不是从最小档重来。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            settings = _settings(freeze_seconds=10, escalate=True, max_freeze_seconds=1000,
                                 escalation_window_seconds=3600)
            manager = _make_manager(config_path, host, store, logger, settings)

            async def freeze_once(event_id: str) -> int:
                await manager.process(
                    [
                        DetectedRateLimit(
                            task="replyer",
                            model="glm-4.6v-flash",
                            status_code=429,
                            source="snapshot",
                            event_id=event_id,
                            detected_at=time.time(),
                        )
                    ]
                )
                entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
                assert entry is not None
                duration = int(entry.until - entry.frozen_at)
                entry.until = time.time() - 1
                await manager.expire_due()
                return duration

            first = await freeze_once("snapshot:esc:1")
            second = await freeze_once("snapshot:esc:2")
            third = await freeze_once("snapshot:esc:3")
            assert (first, second, third) == (10, 20, 40), (first, second, third)

            # 超过判定窗口后重新从最小档开始
            await asyncio.sleep(0.2)
            history = store.state.stats["history"]["replyer\u0000glm-4.6v-flash"]
            stale = time.time() - 7200
            history[0] = stale
            history[2] = stale
            store.save()
            fourth = await freeze_once("snapshot:esc:4")
            assert fourth == 10, fourth

    _run(scenario())


def _load_plugin_module():  # type: ignore[no-untyped-def]
    """按宿主 Runner 的方式加载 plugin.py（需要 maibot_sdk）。"""

    import importlib.util

    plugin_path = PLUGIN_DIR / "plugin.py"
    spec = importlib.util.spec_from_file_location(
        "_maibot_plugin_local_429_model_failover",
        str(plugin_path),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plugin_module_loads() -> None:
    """plugin.py 必须能被宿主方式加载，并导出全部组件（本机缺 SDK 时跳过）。"""

    import importlib.util

    if importlib.util.find_spec("maibot_sdk") is None:
        print("    （跳过：本机没有 maibot_sdk，请在 MaiBot 容器内运行）")
        return

    module = _load_plugin_module()
    plugin = module.create_plugin()
    names = sorted(str(component.get("name")) for component in plugin.get_components())
    assert names == [
        "failover_freeze",
        "failover_help",
        "failover_status",
        "failover_unfreeze",
        "model_freeze_status",
        "unfreeze_model",
    ], names
    assert plugin.get_config_reload_subscriptions() == ["model"]

    config = plugin.build_default_config()
    assert config["freeze"]["freeze_seconds"] == 180
    assert config["detect"]["max_log_age_seconds"] == 300
    assert config["freeze"]["skip_if_no_valid_alternative"] is True
    assert config["detect"]["trigger_status_codes"] == [429]


def test_plugin_reload_confirmation_state_machine() -> None:
    """热重载确认状态机：超时要显示告警，迟到的确认仍要算确认。"""

    import importlib.util
    import logging
    import types

    if importlib.util.find_spec("maibot_sdk") is None:
        print("    （跳过：本机没有 maibot_sdk，请在 MaiBot 容器内运行）")
        return

    module = _load_plugin_module()
    plugin = module.create_plugin()
    plugin._ctx = types.SimpleNamespace(logger=logging.getLogger("test.429-failover"))

    # 刚写完配置 → 等待确认
    plugin._mark_self_write()
    assert "等待中" in plugin._reload_status_line()

    # 超时未确认 → 必须出现告警（修复前这里会被「等待中」挡住）
    plugin._awaiting_reload_at = time.time() - (module.RELOAD_CONFIRM_TIMEOUT_SECONDS + 5)
    plugin._check_reload_confirmation()
    assert plugin._reload_timeout_warned is True
    assert "⚠" in plugin._reload_status_line(), plugin._reload_status_line()

    # 迟到的确认依然要认账，并把告警清掉
    asyncio.run(plugin.on_config_update("model", {}, ""))
    assert plugin._awaiting_reload_at == 0.0
    assert plugin._reload_timeout_warned is False
    assert "正常" in plugin._reload_status_line(), plugin._reload_status_line()

    # 非 model 范围的市场广播不应影响状态
    plugin._awaiting_reload_at = 0.0
    asyncio.run(plugin.on_config_update("bot", {}, ""))
    assert plugin._awaiting_reload_at == 0.0


def test_restore_warns_when_list_edited_manually() -> None:
    """冻结期间有人改过列表时，还原要照做，但要留下明确告警。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            manager = _make_manager(config_path, host, store, logger, _settings(freeze_seconds=60))
            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer",
                        model="glm-4.6v-flash",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:edit:1",
                        detected_at=time.time(),
                    )
                ]
            )
            entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
            assert entry is not None and entry.left_models == ["deepseek-v4-flash"], entry

            # 模拟用户在此期间手工改动列表：删掉一个原本存在的模型，同时把冻结的模型加回来
            host.apply({"replyer": ["glm-4.6v-flash", "embedding-3"]})

            entry.until = time.time() - 1
            _LOGGER_LINES.clear()
            messages = await manager.expire_due()
            assert messages, "应当报告处理结果"
            joined = "\n".join(_LOGGER_LINES)
            assert "被手工改过" in joined, joined
            assert store.state.freezes == []

            # 只新增（没有删除）时应当保留新增并照常还原
            host.apply({"replyer": ["glm-4.6v-flash", "deepseek-v4-flash"]})
            manager2 = _make_manager(config_path, host, store, logger, _settings(freeze_seconds=60))
            await manager2.process(
                [
                    DetectedRateLimit(
                        task="replyer", model="glm-4.6v-flash", status_code=429,
                        source="snapshot", event_id="snapshot:edit:2", detected_at=time.time(),
                    )
                ]
            )
            added = store.state.freeze_for("replyer", "glm-4.6v-flash")
            assert added is not None
            entry = added
            host.apply({"replyer": ["deepseek-v4-flash", "embedding-3", "glm-4.7-flash"]})
            entry.until = time.time() - 1
            await manager2.expire_due()
            lists = host.task_model_lists(host.parse(config_path.read_text(encoding="utf-8")))
            assert "glm-4.6v-flash" in lists["replyer"], lists
            assert "embedding-3" in lists["replyer"] and "glm-4.7-flash" in lists["replyer"], "新增的模型必须保留"

    _run(scenario())


def test_parse_and_format_duration() -> None:
    """可读时长写法解析与格式化。"""

    assert parse_duration("1800") == 1800
    assert parse_duration(600) == 600
    assert parse_duration("30m") == 1800
    assert parse_duration("5h") == 18000
    assert parse_duration("1d") == 86400
    assert parse_duration("1d12h") == 129600
    assert parse_duration("90分钟") == 5400
    assert parse_duration("5 小时") == 18000
    assert parse_duration("2天") == 172800
    assert parse_duration("abc") == 0
    assert parse_duration("") == 0
    assert parse_duration("5x") == 0, "未知单位必须判为非法而不是当成秒"

    assert format_duration(45) == "45s"
    assert format_duration(1800) == "30m"
    assert format_duration(5400) == "1h30m"
    assert format_duration(18000) == "5h"
    assert format_duration(86400) == "1d"
    assert format_duration(129600) == "1d12h"


def test_rule_resolution_precedence_and_overrides() -> None:
    """规则按顺序匹配，先匹配先赢；选项可覆盖全局。"""

    rules, problems = build_rules(
        [
            {"model": "utils-only/*", "task": "utils", "duration": "10m"},
            {"model": "Qwen/*", "duration": "5h"},
            {"model": "*:free", "duration": "1d", "escalate": "off", "max_duration": "2d", "window_duration": "12h"},
            {"model": "broken/*", "duration": "oops"},
            {"model": "", "task": "", "duration": "1h"},
        ]
    )
    assert len(rules) == 3, [rule.describe() for rule in rules]
    assert len(problems) == 2, problems

    plan = resolve_plan(
        "utils", "utils-only/model-a", rules,
        default_seconds=180, default_escalate=True, default_max_seconds=1800, default_window_seconds=900,
    )
    assert plan.seconds == 600 and plan.rule is not None and plan.rule.index == 0
    assert plan.escalate is True and plan.max_seconds == 1800 and plan.window_seconds == 900

    plan = resolve_plan(
        "replyer", "Qwen/Qwen3.7-Flash", rules,
        default_seconds=180, default_escalate=True, default_max_seconds=1800, default_window_seconds=900,
    )
    assert plan.seconds == 18000, plan
    assert plan.rule is not None and plan.rule.index == 1

    plan = resolve_plan(
        "utils", "inclusionai/ling-3.1-flash:free", rules,
        default_seconds=180, default_escalate=True, default_max_seconds=1800, default_window_seconds=900,
    )
    assert plan.seconds == 86400
    assert plan.escalate is False, "规则里写了 off 就不能翻倍"
    assert plan.max_seconds == 172800 and plan.window_seconds == 43200
    assert plan.duration(5) == 86400, "不翻倍时时长固定"

    plan = resolve_plan(
        "utils", "glm-4-flash-250414", rules,
        default_seconds=180, default_escalate=True, default_max_seconds=1800, default_window_seconds=900,
    )
    assert plan.rule is None and plan.seconds == 180, "未命中规则应回到全局默认"
    assert plan.label == "全局默认"
    assert plan.duration(3) == 720, "全局默认仍然翻倍"


def test_rule_duration_used_by_manager() -> None:
    """不同模型命中不同规则时，冻结时长必须分别生效。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            rules, _problems = build_rules(
                [
                    {"model": "deepseek-v4-flash", "duration": "30m"},
                    {"model": "glm-4.6v-flash", "duration": "5h", "escalate": "off"},
                ]
            )
            settings = _settings(freeze_seconds=60, model_rules=tuple(rules))
            manager = _make_manager(config_path, host, store, logger, settings)

            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer", model="deepseek-v4-flash", status_code=429,
                        source="snapshot", event_id="snapshot:rule:1", detected_at=time.time(),
                    )
                ]
            )
            entry = store.state.freeze_for("replyer", "deepseek-v4-flash")
            assert entry is not None and int(entry.until - entry.frozen_at) == 1800, entry

            # 清掉第一条，再冻结另一个模型（它的规则是 5 小时且不翻倍）
            entry.until = time.time() - 1
            await manager.expire_due()
            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer", model="glm-4.6v-flash", status_code=429,
                        source="snapshot", event_id="snapshot:rule:2", detected_at=time.time(),
                    )
                ]
            )
            entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
            assert entry is not None and int(entry.until - entry.frozen_at) == 18000, entry

            # 不翻倍：放回去再被限流，时长仍是 5 小时
            entry.until = time.time() - 1
            await manager.expire_due()
            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer", model="glm-4.6v-flash", status_code=429,
                        source="snapshot", event_id="snapshot:rule:3", detected_at=time.time(),
                    )
                ]
            )
            entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
            assert entry is not None and entry.repeat == 1, entry
            assert int(entry.until - entry.frozen_at) == 18000, entry

            lines = "\n".join(manager.status_lines())
            assert "按模型规则" in lines and "5h" in lines, lines

    _run(scenario())


def test_rule_window_override_controls_escalation() -> None:
    """规则里的判定窗口应覆盖全局窗口。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            rules, _ = build_rules([{"model": "glm-4.6v-flash", "duration": "10s", "window_duration": "1s"}])
            settings = _settings(freeze_seconds=10, escalate=True, max_freeze_seconds=1000,
                                 escalation_window_seconds=3600, model_rules=tuple(rules))
            manager = _make_manager(config_path, host, store, logger, settings)

            async def freeze_once(event_id: str) -> int:
                await manager.process([DetectedRateLimit(
                    task="replyer", model="glm-4.6v-flash", status_code=429,
                    source="snapshot", event_id=event_id, detected_at=time.time(),
                )])
                entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
                assert entry is not None
                duration = int(entry.until - entry.frozen_at)
                entry.until = time.time() - 1
                await manager.expire_due()
                return duration

            assert await freeze_once("snapshot:win:1") == 10
            await asyncio.sleep(1.2)
            assert await freeze_once("snapshot:win:2") == 10, "窗口只有 1 秒，第二次应重新从基础时长开始"

    _run(scenario())


def test_unmatched_rules_reported_by_health_check() -> None:
    """写错通配符/任务名的规则要被体检报出来。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            rules, _ = build_rules(
                [
                    {"model": "no-such-model/*", "duration": "5h"},
                    {"model": "*:free", "task": "no-such-task", "duration": "1d"},
                    {"model": "deepseek-v4-flash", "duration": "30m", "note": "正常规则"},
                ]
            )
            manager = _make_manager(config_path, host, store, logger, _settings(model_rules=tuple(rules)))
            report = manager.health_report()
            assert len(report["unused_rules"]) == 2, report
            lines = "\n".join(manager.health_lines())
            assert "没有匹配到任何现有模型" in lines, lines
            assert "正常规则" not in lines

    _run(scenario())


def test_settings_builder_reads_model_rules() -> None:
    """配置快照要能带上按模型规则（含非法项的报告）。"""

    class Section:
        """简单的配置节替身。"""

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Config:
        """简单的配置替身。"""

        plugin = Section(enabled=True)
        detect = Section()
        freeze = Section(
            model_rules=[
                Section(model="Qwen/*", task="", duration="5h", escalate="inherit",
                        max_duration="", window_duration="", note="5 小时窗口"),
                Section(model="*", task="utils", duration="1d", escalate="off",
                        max_duration="2d", window_duration="12h", note="按天"),
                Section(model="bad/*", task="", duration="???", escalate="inherit",
                        max_duration="", window_duration="", note=""),
            ]
        )
        notify = Section()
        advanced = Section()

    settings = build_settings(Config())
    assert len(settings.model_rules) == 2, settings.model_rules
    assert settings.rule_problems, "非法规则必须被报告"
    plan = settings.freeze_duration_for("replyer", "Qwen/Qwen3.7-Flash", 1)
    assert plan == 18000, plan
    assert settings.plan_for("replyer", "Qwen/Qwen3.7-Flash").rule.note == "5 小时窗口"
    assert settings.freeze_duration_for("utils", "any-model", 3) == 86400, "不翻倍规则"
    assert settings.escalation_window_for("utils", "any-model") == 43200
    assert settings.freeze_duration_for("replyer", "unknown-model", 2) == 360, "回到全局默认并翻倍"
    assert settings.plan_lines() and "按模型规则" in settings.plan_lines()[0]


def test_restore_respects_manual_removal() -> None:
    """冻结期间用户把模型从列表里删掉后，到期不应把它插回去。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            manager = _make_manager(config_path, host, store, logger, _settings(freeze_seconds=60))
            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer", model="glm-4.6v-flash", status_code=429,
                        source="snapshot", event_id="snapshot:manual:1", detected_at=time.time(),
                    )
                ]
            )
            entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
            assert entry is not None and entry.left_models == ["deepseek-v4-flash"]

            # 用户在此期间重做列表：删掉了我们离开时列表里的模型，且被冻结的模型没有回来
            host.apply({"replyer": ["embedding-3"]})
            user_text = config_path.read_text(encoding="utf-8")

            entry.until = time.time() - 1
            _LOGGER_LINES.clear()
            messages = await manager.expire_due()
            assert any("未还原" in item for item in messages), messages
            assert config_path.read_text(encoding="utf-8") == user_text, "不得覆盖用户的手工编辑"
            assert store.state.freezes == []

            # 关掉该行为后同样场景会强制还原（先把模型加回列表再冻结）
            host.apply({"replyer": ["glm-4.6v-flash", "deepseek-v4-flash", "embedding-3"]})
            settings = _settings(freeze_seconds=60, respect_manual_edit=False)
            manager2 = _make_manager(config_path, host, store, logger, settings)
            await manager2.process(
                [
                    DetectedRateLimit(
                        task="replyer", model="glm-4.6v-flash", status_code=429,
                        source="snapshot", event_id="snapshot:manual:2", detected_at=time.time(),
                    )
                ]
            )
            entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
            assert entry is not None
            host.apply({"replyer": ["deepseek-v4-flash", "embedding-3"]})
            entry.until = time.time() - 1
            await manager2.expire_due()
            lists = host.task_model_lists(host.parse(config_path.read_text(encoding="utf-8")))
            assert "glm-4.6v-flash" in lists["replyer"], lists

    _run(scenario())


def test_frozen_model_counts_as_existing_for_rule_check() -> None:
    """被自己冻结的模型不能算“规则没匹配到”——否则体检会每 10 分钟误报一次。"""

    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            config_path, host, store, logger = _make_workspace(tmp)
            store.load()
            rules, _ = build_rules([{"model": "glm-4.6v-flash", "duration": "5h"}])
            manager = _make_manager(config_path, host, store, logger, _settings(model_rules=tuple(rules)))

            assert manager.health_report()["unused_rules"] == [], "冻结前不该报未命中"

            await manager.process(
                [
                    DetectedRateLimit(
                        task="replyer",
                        model="glm-4.6v-flash",
                        status_code=429,
                        source="snapshot",
                        event_id="snapshot:rule-health:1",
                        detected_at=time.time(),
                    )
                ]
            )
            entry = store.state.freeze_for("replyer", "glm-4.6v-flash")
            assert entry is not None, "应当已冻结"

            report = manager.health_report()
            assert report["unused_rules"] == [], f"冻结中的模型不应被判为规则未命中: {report}"
            assert manager.health_lines() == [], manager.health_lines()

            # 还原后依然不报
            entry.until = time.time() - 1
            await manager.expire_due()
            assert manager.health_report()["unused_rules"] == []

    _run(scenario())


def test_touch_config_keeps_content_and_inode() -> None:
    """touch_config 必须只“碰一下”文件：内容与 inode 都不变，但走一遍写入路径。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        config_path, host, store, logger = _make_workspace(tmp)
        store.load()
        writes: list = []
        manager = FreezeManager(
            host_config=host,
            store=store,
            logger=logger,
            settings_provider=lambda: _settings(),
            backup_dir=config_path.parent / "backups",
            on_write=lambda: writes.append(time.time()),
        )
        before_text = config_path.read_text(encoding="utf-8")
        before_inode = config_path.stat().st_ino

        assert manager.touch_config() is True
        assert config_path.read_text(encoding="utf-8") == before_text, "内容必须完全不变"
        assert config_path.stat().st_ino == before_inode, "不得更换 inode（宿主监听挂在 inode 上）"
        assert len(writes) == 1, "必须标记为「本插件写入」，避免被当成外部改动"


def test_plugin_startup_grace_retries_instead_of_alarming() -> None:
    """刚加载时收不到热重载回执：宽限期内按间隔反复重写配置，只有超出宽限期才告警。"""

    import importlib.util
    import logging
    import types

    if importlib.util.find_spec("maibot_sdk") is None:
        print("    （跳过：本机没有 maibot_sdk，请在 MaiBot 容器内运行）")
        return

    module = _load_plugin_module()
    plugin = module.create_plugin()
    plugin._ctx = types.SimpleNamespace(logger=logging.getLogger("test.429-failover-2"))

    touched: list = []
    plugin._manager = types.SimpleNamespace(touch_config=lambda: (touched.append(time.time()), True)[1])

    timeout = module.RELOAD_CONFIRM_TIMEOUT_SECONDS
    interval = module.RELOAD_RETRY_INTERVAL_SECONDS

    # 场景一：启动宽限期内超时 → 重写一次配置、不告警
    plugin._loaded_at = time.time() - 10
    plugin._mark_self_write()
    plugin._awaiting_reload_at = time.time() - (timeout + 1)
    plugin._check_reload_confirmation()
    assert len(touched) == 1, "启动宽限期内应当重写配置以触发热重载"
    assert plugin._reload_timeout_warned is False, "宽限期内不应进入「监听失效」告警状态"
    assert plugin._awaiting_reload_at > time.time() - 5, "应当重新开始等待确认"
    assert "⚠" not in plugin._reload_status_line(), plugin._reload_status_line()

    # 场景二：还不到重试间隔 → 不重复重写，也不告警
    plugin._awaiting_reload_at = time.time() - (timeout + 1)
    plugin._check_reload_confirmation()
    assert len(touched) == 1, "未到重试间隔不应重复重写"
    assert plugin._reload_timeout_warned is False

    # 场景三：到了重试间隔 → 再重写一次，仍然不告警
    plugin._reload_retry_at = time.time() - 1
    plugin._awaiting_reload_at = time.time() - (timeout + 1)
    plugin._check_reload_confirmation()
    assert len(touched) == 2, "到间隔后应再次重写"
    assert plugin._reload_timeout_warned is False

    # 场景四：超出启动宽限期仍无回执 → 才进入告警状态
    plugin._loaded_at = time.time() - (module.STARTUP_GRACE_SECONDS + 60)
    plugin._reload_retry_at = 0.0
    plugin._awaiting_reload_at = time.time() - (timeout + 1)
    before = len(touched)
    plugin._check_reload_confirmation()
    assert len(touched) == before, "超出宽限期不应再重写"
    assert plugin._reload_timeout_warned is True
    assert "⚠" in plugin._reload_status_line(), plugin._reload_status_line()

    # 场景五：迟到的确认依然认账，并清掉告警
    asyncio.run(plugin.on_config_update("model", {}, ""))
    assert plugin._awaiting_reload_at == 0.0 and plugin._reload_timeout_warned is False
    assert "⚠" not in plugin._reload_status_line()


def test_state_persistence() -> None:
    """状态文件应可跨进程重启保留冻结记录。"""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        logger = ListLogger()
        store = StateStore(tmp / "state.json", logger)
        store.load()
        from failover.state import FreezeEntry

        store.state.add_freeze(
            FreezeEntry(
                task="replyer",
                model="glm-4.6v-flash",
                index=0,
                scope="task",
                frozen_at=time.time(),
                until=time.time() + 60,
                reason="HTTP 429（snapshot）",
            )
        )
        store.bump("detected")
        store.save()

        reloaded = StateStore(tmp / "state.json", logger)
        state = reloaded.load()
        assert state.freeze_for("replyer", "glm-4.6v-flash") is not None
        assert int(state.stats.get("detected", 0)) == 1


def test_settings_builder() -> None:
    """配置快照构造应正确处理缺省值与排除项。"""

    class Section:
        """简单的配置节替身。"""

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Config:
        """简单的配置替身。"""

        plugin = Section(enabled=True, admin_user_ids=["10001"])
        detect = Section(poll_interval_seconds=0.1, trigger_status_codes=[429, "500", "bad"])
        freeze = Section(freeze_scope="MODEL", max_freeze_seconds=0)
        notify = Section()
        advanced = Section()

    settings = build_settings(Config())
    assert settings.status_codes == (429, 500)
    assert settings.poll_interval_seconds == 0.5
    assert settings.global_scope is True
    assert settings.is_admin("10001") and not settings.is_admin("10002")
    assert settings.is_admin("", is_local_operator=True)
    assert settings.max_freeze_seconds >= 1


def test_host_paths_probe() -> None:
    """宿主根目录探测：本机存在 /MaiMBot 时应能命中。"""

    from failover import host_paths

    root = host_paths.resolve_host_root()
    if Path("/MaiMBot/config/model_config.toml").is_file():
        assert root is not None and (root / "config" / "model_config.toml").is_file()
        assert host_paths.resolve_llm_error_dir() is not None
    else:
        print(f"    （跳过：本机未探测到宿主根目录，返回 {root}）")


def main() -> int:
    """运行全部测试。"""

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures: List[str] = []
    for name, func in tests:
        try:
            func()
        except Exception as exc:  # noqa: BLE001 - 测试入口需要收集全部失败
            failures.append(f"{name}: {exc}")
            print(f"[FAIL] {name}: {exc}")
            with contextlib.suppress(Exception):
                import traceback

                traceback.print_exc()
        else:
            print(f"[ OK ] {name}")

    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} 个测试失败")
        return 1
    print(f"全部 {len(tests)} 个测试通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

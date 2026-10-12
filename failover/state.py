"""插件持久化状态：冻结记录、已处理事件、累计统计。

状态文件位于插件数据目录（``ctx.paths.data_dir/state.json``），宿主侧可见于
``data/MaiMBot-plugin-data/<插件id>/``，便于排障与手工干预。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import contextlib
import json
import os
import tempfile
import time

STATE_VERSION = 1
MAX_PROCESSED_EVENTS = 800
"""最多记住多少条已处理事件 ID，用于跨重启去重。"""


@dataclass
class FreezeEntry:
    """一条冻结记录。

    ``index`` / ``after_model`` / ``before_model`` 都是“移除时”列表里的位置信息，
    用于还原时把模型插回原来的相对位置：优先以原本的后继模型（``before_model``）
    为锚点插入其前，其次以原本的前驱模型（``after_model``）为锚点插入其后，
    两个锚点都不在列表里时退回使用 ``index``。
    """

    task: str
    model: str
    index: int
    scope: str
    frozen_at: float
    until: float
    repeat: int = 1
    reason: str = ""
    applied: bool = True
    status_code: int = 0
    source: str = ""
    after_model: str = ""
    before_model: str = ""
    array_style: Dict[str, Any] = field(default_factory=dict)
    """冻结时该任务 ``model_list`` 的排版风格，用于还原成原样。"""
    left_models: List[str] = field(default_factory=list)
    """本插件改写后该任务 ``model_list`` 的样子；用于发现冻结期间有人在手工改配置。"""
    context_models: List[str] = field(default_factory=list)
    """冻结那一刻该任务 ``model_list`` 的完整顺序（含本模型）。

    还原时用它推导“前驱/后继锚点”：同一个模型可能先被别人挡住（列表缩短），
    只有更早的上下文里才保留了它与其它模型的相对位置。
    """

    @property
    def key(self) -> str:
        """冻结记录的唯一键。"""

        return f"{self.task}\u0000{self.model}"

    def remaining(self, now: Optional[float] = None) -> float:
        """返回剩余冻结秒数（负数表示已到期）。"""

        return self.until - (time.time() if now is None else now)

    def describe_remaining(self, now: Optional[float] = None) -> str:
        """以中文描述剩余冻结时间。"""

        remaining = max(0, int(self.remaining(now)))
        if remaining >= 3600:
            return f"{remaining // 3600} 小时 {remaining % 3600 // 60} 分"
        if remaining >= 60:
            return f"{remaining // 60} 分 {remaining % 60} 秒"
        return f"{remaining} 秒"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "FreezeEntry":
        """从字典还原（容忍缺字段）。"""

        return cls(
            task=str(raw.get("task", "")),
            model=str(raw.get("model", "")),
            index=int(raw.get("index", 0)),
            scope=str(raw.get("scope", "task")),
            frozen_at=float(raw.get("frozen_at", 0.0)),
            until=float(raw.get("until", 0.0)),
            repeat=int(raw.get("repeat", 1)),
            reason=str(raw.get("reason", "")),
            applied=bool(raw.get("applied", True)),
            status_code=int(raw.get("status_code", 0) or 0),
            source=str(raw.get("source", "")),
            after_model=str(raw.get("after_model", "")),
            before_model=str(raw.get("before_model", "")),
            array_style=dict(raw.get("array_style")) if isinstance(raw.get("array_style"), dict) else {},
            left_models=[str(item) for item in raw.get("left_models", [])]
            if isinstance(raw.get("left_models"), list)
            else [],
            context_models=[str(item) for item in raw.get("context_models", [])]
            if isinstance(raw.get("context_models"), list)
            else [],
        )


@dataclass
class PluginState:
    """插件整体状态。"""

    freezes: List[FreezeEntry] = field(default_factory=list)
    processed_events: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def freeze_for(self, task: str, model: str) -> Optional[FreezeEntry]:
        """查找指定任务与模型的冻结记录。"""

        for entry in self.freezes:
            if entry.task == task and entry.model == model:
                return entry
        return None

    def add_freeze(self, entry: FreezeEntry) -> None:
        """写入或覆盖一条冻结记录。"""

        self.freezes = [item for item in self.freezes if item.key != entry.key]
        self.freezes.append(entry)

    def remove_freeze(self, task: str, model: str) -> Optional[FreezeEntry]:
        """删除一条冻结记录并返回它。"""

        for index, entry in enumerate(self.freezes):
            if entry.task == task and entry.model == model:
                return self.freezes.pop(index)
        return None


class StateStore:
    """``state.json`` 的读写器（原子写入）。"""

    def __init__(self, path: Path, logger: Any) -> None:
        """初始化状态仓库。

        Args:
            path: ``state.json`` 路径。
            logger: 插件日志器。
        """

        self.path = Path(path)
        self._logger = logger
        self._processed: "OrderedDict[str, None]" = OrderedDict()
        self.state = PluginState()

    # ------------------------------------------------------------------ 读取

    def load(self) -> PluginState:
        """读取状态文件；文件缺失或损坏时返回空状态。"""

        if not self.path.is_file():
            self.state = PluginState()
            return self.state

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning(f"状态文件读取失败，将以空状态启动: {exc}")
            self.state = PluginState()
            return self.state

        freezes: List[FreezeEntry] = []
        if isinstance(raw.get("freezes"), list):
            for item in raw["freezes"]:
                if isinstance(item, dict):
                    with contextlib.suppress(TypeError, ValueError):
                        freezes.append(FreezeEntry.from_dict(item))

        processed_raw = raw.get("processed_events")
        processed: List[str] = []
        if isinstance(processed_raw, list):
            processed = [str(item) for item in processed_raw][-MAX_PROCESSED_EVENTS:]

        stats = raw.get("stats") if isinstance(raw.get("stats"), dict) else {}
        self.state = PluginState(freezes=freezes, processed_events=processed, stats=dict(stats))
        self._processed = OrderedDict.fromkeys(processed)
        return self.state

    # ------------------------------------------------------------------ 写入

    def mark_processed(self, event_id: str) -> bool:
        """登记事件 ID；返回 ``True`` 表示这是首次见到（需要处理）。"""

        if not event_id:
            return True
        if event_id in self._processed:
            return False
        self._processed[event_id] = None
        while len(self._processed) > MAX_PROCESSED_EVENTS:
            self._processed.popitem(last=False)
        self.state.processed_events = list(self._processed.keys())
        return True

    def bump(self, key: str, delta: int = 1) -> int:
        """累加统计计数并返回新值。"""

        value = int(self.state.stats.get(key, 0) or 0) + delta
        self.state.stats[key] = value
        return value

    def note(self, key: str, value: Any) -> None:
        """记录一条最近事件信息。"""

        self.state.stats[key] = value

    def save(self) -> None:
        """原子写回状态文件。"""

        payload = {
            "version": STATE_VERSION,
            "freezes": [entry.to_dict() for entry in self.state.freezes],
            "processed_events": list(self._processed.keys()),
            "stats": self.state.stats,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, self.path)
                tmp_name = ""
            finally:
                if tmp_name:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_name)
        except OSError as exc:
            self._logger.warning(f"状态文件写入失败: {exc}")

    def purge_processed(self, keep: Iterable[str] = ()) -> None:
        """清空事件去重记录（保留指定项）。"""

        self._processed = OrderedDict.fromkeys(str(item) for item in keep if item)
        self.state.processed_events = list(self._processed.keys())

"""限流（429）事件检测。

插件无法挂到宿主 LLM 调用链上，所以改用宿主自己产出的两份“地面真相”：

1. **LLM 失败快照**（首选）：``logs/maisaka_prompt/llm_error/<会话>/<时间戳>.json``。
   宿主每次模型尝试失败都会写/更新快照，里面有结构化的
   ``request.task_name``、``generation_attempts[].model`` 与
   ``generation_attempts[].error.status_code``，字段稳定、无需正则。
2. **应用日志**（兜底）：``logs/app_*.log.jsonl`` 里 ``llm_models`` 记录器的
   告警/错误行，例如
   ``任务 'utils' 的模型 'inclusionai/ling-3.1-flash:free' 遇到可重试的HTTP错误: ... Error code: 429``。

两条通道都只读文件，不触碰宿主进程；任一通道报错都不会影响另一条。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import contextlib
import hashlib
import json
import re
import time

SNAPSHOT_SOURCE = "snapshot"
LOG_SOURCE = "log"

MAX_SNAPSHOT_FILES_PER_SCAN = 300
MAX_SNAPSHOT_FILE_SIZE = 4 * 1024 * 1024
MAX_LOG_FILES_PER_SCAN = 3
LOG_TAIL_BYTES_ON_FIRST_SIGHT = 256 * 1024
SCAN_OVERLAP_SECONDS = 10.0
DEFAULT_MAX_LOG_EVENT_AGE_SECONDS = 300.0
"""日志通道的默认最大事件年龄（秒）：避免首次挂载时把几百行以前的历史 429 当成刚发生的事。"""

TASK_MODEL_RE = re.compile(r"任务[ ]['\"](?P<task>[^'\"]+)['\"] 的模型[ ]['\"](?P<model>[^'\"]+)['\"]")
STATUS_CODE_RES: Tuple[re.Pattern[str], ...] = (
    re.compile(r"遇到[ ](?P<code>\d{3})[ ]错误"),
    re.compile(r"Error code:[ ]*(?P<code>\d{3})"),
    re.compile(r"HTTP[ ](?P<code>\d{3})"),
    re.compile(r"status_code[\"']?[:=][ ]*(?P<code>\d{3})"),
)
RETRYABLE_HTTP_MARKERS = ("遇到可重试的HTTP错误", "遇到可重试的 HTTP 错误")
FINAL_FAILURE_MARKERS = ("用尽重试次数后仍然失败", "重试用尽后仍然失败")
RATE_LIMIT_MARKERS = (
    "429",
    "RateLimitError",
    "rate_limit_error",
    "请求过于频繁",
    "too many requests",
    "temporarily unavailable",
)


@dataclass(frozen=True)
class DetectedRateLimit:
    """一条检测到的限流事件。"""

    task: str
    model: str
    status_code: int
    source: str
    event_id: str
    detected_at: float
    detail: str = ""

    def describe(self) -> str:
        """生成可读描述。"""

        code = self.status_code or "限流"
        return f"[{self.source}] 任务 {self.task} 的模型 {self.model} 返回 {code}"


class FailureDetector:
    """扫描宿主失败快照与日志，产出限流事件。"""

    def __init__(
        self,
        *,
        logger: Any,
        error_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        status_codes: Iterable[int] = (429,),
        detect_rate_limit_by_type: bool = True,
        enable_snapshot_scan: bool = True,
        enable_log_scan: bool = True,
        window_seconds: float = 30.0,
        max_log_event_age_seconds: float = DEFAULT_MAX_LOG_EVENT_AGE_SECONDS,
    ) -> None:
        """初始化检测器。

        Args:
            logger: 插件日志器。
            error_dir: LLM 失败快照目录。
            log_dir: 宿主日志目录。
            status_codes: 触发冻结的状态码集合。
            detect_rate_limit_by_type: 状态码缺失时是否按限流特征（RateLimitError 等）判定。
            enable_snapshot_scan: 是否启用快照扫描。
            enable_log_scan: 是否启用日志扫描。
            window_seconds: 单轮扫描最多回溯的时间窗口（秒）。
            max_log_event_age_seconds: 日志行允许的最大年龄（秒），超过则丢弃；``0`` 表示不限制。
        """

        self._logger = logger
        self.error_dir = Path(error_dir) if error_dir else None
        self.log_dir = Path(log_dir) if log_dir else None
        self.status_codes: Set[int] = {int(code) for code in status_codes}
        self.detect_rate_limit_by_type = bool(detect_rate_limit_by_type)
        self.enable_snapshot_scan = bool(enable_snapshot_scan)
        self.enable_log_scan = bool(enable_log_scan)
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_log_event_age_seconds = max(0.0, float(max_log_event_age_seconds))

        self._last_scan_at: float = 0.0
        self._log_offsets: Dict[str, int] = {}
        self._log_inodes: Dict[str, int] = {}
        self._stale_log_skipped: int = 0
        """因过旧被丢弃的日志事件数量（用于排查与展示）。"""

    # ------------------------------------------------------------------ 对外

    def describe_sources(self, show_paths: bool = True) -> str:
        """描述当前数据来源可用性。

        Args:
            show_paths: 是否回显宿主侧的绝对路径。``False`` 时只给出可用性，
                不回显目录 —— ``/429状态`` 对所有人开放，默认不该把服务器的
                目录布局发进聊天。
        """

        def describe(label: str, path: object, ok: bool) -> str:
            """生成一行来源描述。"""

            if show_paths:
                return f"{label} {'可用' if ok else '不可用'}（{path or '未解析'}）"
            return f"{label} {'可用' if ok else '不可用'}（宿主路径仅管理员可见）"

        snapshot_ok = bool(self.error_dir and self.error_dir.is_dir())
        log_ok = bool(self.log_dir and self.log_dir.is_dir())
        parts = [
            describe("快照目录", self.error_dir, snapshot_ok),
            describe("日志目录", self.log_dir, log_ok),
        ]
        if self._stale_log_skipped:
            parts.append(f"已丢弃过旧日志事件 {self._stale_log_skipped} 条")
        return "；".join(parts)

    def scan(self, now: Optional[float] = None) -> List[DetectedRateLimit]:
        """扫描一次，返回本轮检测到的限流事件。"""

        current = time.time() if now is None else now
        since = self._last_scan_at - SCAN_OVERLAP_SECONDS if self._last_scan_at else current - self.window_seconds
        self._last_scan_at = current

        detections: List[DetectedRateLimit] = []
        if self.enable_snapshot_scan:
            detections.extend(self._scan_snapshots(since, current))
        if self.enable_log_scan:
            detections.extend(self._scan_logs(current))
        return detections

    # ------------------------------------------------------------------ 快照

    def _iter_snapshot_files(self, since: float) -> List[Path]:
        """列出需要解析的快照文件（按修改时间倒序）。"""

        if self.error_dir is None or not self.error_dir.is_dir():
            return []

        candidates: List[Tuple[float, Path]] = []
        try:
            for path in self.error_dir.rglob("*.json"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_size <= 0 or stat.st_size > MAX_SNAPSHOT_FILE_SIZE:
                    continue
                if stat.st_mtime < since:
                    continue
                candidates.append((stat.st_mtime, path))
        except OSError as exc:
            self._logger.debug(f"扫描失败快照目录出错: {exc}")
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [path for _mtime, path in candidates[:MAX_SNAPSHOT_FILES_PER_SCAN]]

    def _scan_snapshots(self, since: float, now: float) -> List[DetectedRateLimit]:
        """解析失败快照，提取限流尝试。"""

        detections: List[DetectedRateLimit] = []
        for path in self._iter_snapshot_files(since):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._logger.debug(f"解析失败快照 {path.name} 出错: {exc}")
                continue
            if not isinstance(payload, dict):
                continue
            detections.extend(self._extract_from_snapshot(payload, path, now))
        return detections

    def _extract_from_snapshot(
        self,
        payload: Dict[str, Any],
        path: Path,
        now: float,
    ) -> List[DetectedRateLimit]:
        """从单份快照里提取限流尝试。"""

        request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        task = str(request.get("task_name") or "").strip()
        default_model = str(metadata.get("model_name") or "").strip()
        attempts = payload.get("generation_attempts")
        if not isinstance(attempts, list):
            return []

        results: List[DetectedRateLimit] = []
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                continue
            error = attempt.get("error") if isinstance(attempt.get("error"), dict) else {}
            if not error:
                continue
            status_code = self._coerce_status(error.get("status_code"))
            message = str(error.get("message") or "")
            error_type = str(error.get("type") or "")
            if not self._is_target_error(status_code, error_type, message):
                continue

            model = str(attempt.get("model") or default_model).strip()
            attempt_id = str(attempt.get("attempt_id") or f"{path.stem}#{index}")
            if not task or not model:
                continue

            results.append(
                DetectedRateLimit(
                    task=task,
                    model=model,
                    status_code=status_code or 429,
                    source=SNAPSHOT_SOURCE,
                    event_id=f"snapshot:{attempt_id}",
                    detected_at=now,
                    detail=self._shorten(message),
                )
            )
        return results

    # ------------------------------------------------------------------ 日志

    def _iter_log_files(self) -> List[Path]:
        """列出最近的宿主应用日志文件。"""

        if self.log_dir is None or not self.log_dir.is_dir():
            return []
        candidates: List[Tuple[float, Path]] = []
        try:
            for path in self.log_dir.glob("app_*.log.jsonl"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_size <= 0:
                    continue
                candidates.append((stat.st_mtime, path))
        except OSError as exc:
            self._logger.debug(f"扫描宿主日志目录出错: {exc}")
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [path for _mtime, path in candidates[:MAX_LOG_FILES_PER_SCAN]]

    def _scan_logs(self, now: float) -> List[DetectedRateLimit]:
        """增量读取宿主日志，提取限流告警。"""

        detections: List[DetectedRateLimit] = []
        for path in self._iter_log_files():
            detections.extend(self._scan_single_log(path, now))
        self._forget_stale_offsets()
        return detections

    def _scan_single_log(self, path: Path, now: float) -> List[DetectedRateLimit]:
        """读取单个日志文件的新增内容。"""

        key = str(path)
        try:
            stat = path.stat()
        except OSError:
            return []

        inode = getattr(stat, "st_ino", 0)
        offset = self._log_offsets.get(key)
        if offset is None or self._log_inodes.get(key) != inode or stat.st_size < offset:
            offset = max(0, stat.st_size - LOG_TAIL_BYTES_ON_FIRST_SIGHT)

        results: List[DetectedRateLimit] = []
        final_offset = offset
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                position = offset
                for raw_line in handle:
                    if not raw_line.endswith(b"\n"):
                        # 写入中途的半行：本轮不消费，留到下一轮再读
                        break
                    line = raw_line.decode("utf-8", errors="replace")
                    position += len(raw_line)
                    parsed = self._parse_log_line(line)
                    if parsed is None:
                        continue
                    task, model, status_code, detail, event_time = parsed
                    if not self._is_fresh_enough(event_time, now):
                        self._stale_log_skipped += 1
                        continue
                    results.append(
                        DetectedRateLimit(
                            task=task,
                            model=model,
                            status_code=status_code,
                            source=LOG_SOURCE,
                            event_id=self._log_event_id(key, position - len(raw_line), line),
                            detected_at=now,
                            detail=detail,
                        )
                    )
                final_offset = position
        except OSError as exc:
            self._logger.debug(f"读取宿主日志 {path.name} 出错: {exc}")
            return results

        self._log_offsets[key] = final_offset
        self._log_inodes[key] = inode
        return results

    @staticmethod
    def _log_event_id(file_key: str, line_start: int, line: str) -> str:
        """构造日志事件 ID（文件 + 偏移 + 内容摘要）。"""

        digest = hashlib.sha1(line.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"log:{Path(file_key).name}:{line_start}:{digest}"

    def _parse_log_line(self, line: str) -> Optional[Tuple[str, str, int, str, Optional[float]]]:
        """解析一行日志，命中限流时返回 ``(任务, 模型, 状态码, 详情, 事件时间)``。"""

        stripped = line.strip()
        if not stripped or "任务" not in stripped:
            return None
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(record, dict):
            return None

        logger_name = str(record.get("logger_name") or record.get("logger") or "")
        if logger_name and "llm" not in logger_name.lower():
            return None

        event = str(record.get("event") or "")
        if not event:
            return None
        if not any(marker in event for marker in (*RETRYABLE_HTTP_MARKERS, *FINAL_FAILURE_MARKERS)):
            return None

        match = TASK_MODEL_RE.search(event)
        if match is None:
            return None

        status_code = self._extract_status_code(event)
        if not self._is_target_error(status_code, "", event):
            return None

        event_time = self._parse_event_time(record.get("timestamp"))
        return match.group("task"), match.group("model"), status_code or 429, self._shorten(event), event_time

    # ------------------------------------------------------------------ 工具

    def _is_fresh_enough(self, event_time: Optional[float], now: float) -> bool:
        """判断日志事件是否足够新。

        宿主的日志文件会一直保留（每个进程一个 ``app_*.jsonl``），插件首次挂载、
        重建读取偏移、或日志轮转后重新定位时，可能读到几百行以前的历史 429。
        这里用日志行自带的时间戳兜底：解析不出来时按“新鲜”处理（宁可多处理一次，
        也不要漏掉真实事件）。
        """

        if self.max_log_event_age_seconds <= 0 or event_time is None:
            return True
        return (now - event_time) <= self.max_log_event_age_seconds

    @staticmethod
    def _parse_event_time(raw: Any) -> Optional[float]:
        """解析日志行里的 ``timestamp`` 字段。

        宿主业务日志是 ``MM-DD HH:MM:SS``（本地时间，无年份），插件日志是
        ``YYYY-MM-DDTHH:MM:SS.ffffffZ``（UTC）。年份缺失时用当前年份推断，
        并处理 12 月/1 月的跨年边界。
        """

        text = str(raw or "").strip()
        if not text:
            return None

        if "T" in text:
            with contextlib.suppress(ValueError):
                parsed = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
                return parsed.replace(tzinfo=timezone.utc).timestamp()
            return None

        with contextlib.suppress(ValueError):
            now = datetime.now()
            parsed = datetime.strptime(f"{now.year}-{text[:14]}", "%Y-%m-%d %H:%M:%S")
            if parsed - now > timedelta(days=1):
                parsed = parsed.replace(year=now.year - 1)
            return parsed.timestamp()
        return None

    def _forget_stale_offsets(self) -> None:
        """清理已不存在日志文件的读取偏移。"""

        for key in list(self._log_offsets.keys()):
            if not Path(key).is_file():
                self._log_offsets.pop(key, None)
                self._log_inodes.pop(key, None)

    @staticmethod
    def _coerce_status(value: Any) -> int:
        """把状态码字段转成整数。"""

        with contextlib.suppress(TypeError, ValueError):
            if value is not None:
                return int(value)
        return 0

    @classmethod
    def _extract_status_code(cls, text: str) -> int:
        """从文本里提取 HTTP 状态码。"""

        for pattern in STATUS_CODE_RES:
            match = pattern.search(text)
            if match is not None:
                return cls._coerce_status(match.group("code"))
        return 0

    def _is_target_error(self, status_code: int, error_type: str, message: str) -> bool:
        """判断是否属于需要冻结的错误。"""

        if status_code and status_code in self.status_codes:
            return True
        if not self.detect_rate_limit_by_type:
            return False
        haystack = f"{error_type} {message}"
        return any(marker.lower() in haystack.lower() for marker in RATE_LIMIT_MARKERS)

    @staticmethod
    def _shorten(text: str, limit: int = 300) -> str:
        """压缩日志/错误文本用于展示。"""

        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit]}…"


def normalize_status_codes(raw: Sequence[Any]) -> List[int]:
    """把配置里的状态码列表规范化为去重整数列表。"""

    result: List[int] = []
    for item in raw:
        with contextlib.suppress(TypeError, ValueError):
            code = int(str(item).strip())
            if 100 <= code <= 599 and code not in result:
                result.append(code)
    return result or [429]

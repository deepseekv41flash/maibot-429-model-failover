"""按模型/任务自定义冻结时长的规则解析。

不同上游的限流窗口差别很大：有的按分钟、有的按 5 小时、有的干脆按天。
全局的 ``freeze_seconds`` 只能给一个折中值，所以这里引入「规则」：

* 规则由插件配置页的列表维护（``freeze.model_rules``），每条包含
  ``model``（支持 ``*`` 通配）、可选 ``task``（限定模型类别）、
  ``duration``（冻结时长，可读写法：``30m`` / ``5h`` / ``1d`` / ``1800``），
  以及可选的 ``escalate`` / ``max_duration`` / ``window_duration`` 覆盖项；
* 命中顺序 = 配置顺序，**先匹配先赢**，所以把更具体的规则写在前面即可；
* 所有规则都不命中时退回全局默认（``freeze_seconds`` / ``max_freeze_seconds`` /
  ``escalation_window_seconds``）。

模块本身不依赖 SDK，规则可以用普通对象或字典构造，方便离线测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import re

ESCALATE_INHERIT = "inherit"
ESCALATE_ON = "on"
ESCALATE_OFF = "off"

_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "秒": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "分": 60,
    "分钟": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "时": 3600,
    "小时": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
    "天": 86400,
    "日": 86400,
}

_DURATION_TOKEN_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z\u4e00-\u9fff]*)")


def parse_duration(text: Any, default: int = 0) -> int:
    """解析可读时长写法。

    支持 ``1800``（秒）、``30m``、``5h``、``1d``、``1d12h``、``90分钟``、``5 小时``。

    Args:
        text: 待解析的文本。
        default: 解析失败时返回的默认值。

    Returns:
        int: 秒数；解析失败返回 ``default``。
    """

    if isinstance(text, (int, float)):
        value = int(text)
        return value if value > 0 else default

    raw = str(text or "").strip().lower().replace(" ", "")
    if not raw:
        return default

    total = 0.0
    matched_any = False
    matched_all = True
    for token in _DURATION_TOKEN_RE.finditer(raw):
        value = float(token.group("value"))
        unit = token.group("unit")
        if not unit:
            total += value
        elif unit in _UNIT_SECONDS:
            total += value * _UNIT_SECONDS[unit]
        else:
            matched_all = False
            continue
        matched_any = True

    if not matched_any or not matched_all:
        return default
    seconds = int(round(total))
    return seconds if seconds > 0 else default


def format_duration(seconds: int) -> str:
    """把秒数格式化成易读文本（1d12h / 5h / 30m / 45s）。"""

    remaining = max(0, int(seconds))
    if remaining == 0:
        return "0s"
    parts: List[str] = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if remaining >= size:
            count, remaining = divmod(remaining, size)
            parts.append(f"{count}{unit}")
    if remaining:
        parts.append(f"{remaining}s")
    return "".join(parts)


@dataclass(frozen=True)
class FreezeRule:
    """解析后的规则。"""

    index: int
    model_pattern: str
    task: str
    seconds: int
    escalate: Optional[bool]
    max_seconds: Optional[int]
    window_seconds: Optional[int]
    note: str = ""

    def matches(self, task: str, model: str) -> bool:
        """判断该规则是否命中给定的任务与模型。"""

        if self.task and self.task != task:
            return False
        pattern = self.model_pattern or "*"
        return fnmatch(model, pattern)

    def label(self) -> str:
        """规则简称，用于日志与状态展示。"""

        scope = f"[{self.task}] " if self.task else ""
        return f"{scope}{self.model_pattern or '*'}"

    def describe(self) -> str:
        """规则完整描述。"""

        extras: List[str] = []
        if self.escalate is False:
            extras.append("不翻倍")
        elif self.escalate is True:
            extras.append("翻倍")
        if self.max_seconds is not None:
            extras.append(f"上限 {format_duration(self.max_seconds)}")
        if self.window_seconds is not None:
            extras.append(f"判定窗口 {format_duration(self.window_seconds)}")
        if self.note:
            extras.append(self.note)
        suffix = f"（{'，'.join(extras)}）" if extras else ""
        return f"#{self.index + 1} {self.label()} → {format_duration(self.seconds)}{suffix}"


@dataclass(frozen=True)
class FreezePlan:
    """针对某个 (任务, 模型) 解析出的冻结计划。"""

    seconds: int
    escalate: bool
    max_seconds: int
    window_seconds: int
    rule: Optional[FreezeRule] = None

    @property
    def source(self) -> str:
        """计划来源描述（命中哪条规则或全局默认）。"""

        return f"规则 {self.rule.describe()}" if self.rule is not None else "全局默认"

    @property
    def label(self) -> str:
        """简短来源标签，用于日志/状态行。"""

        return f"规则 #{self.rule.index + 1} {self.rule.label()}" if self.rule is not None else "全局默认"

    def duration(self, repeat: int) -> int:
        """按重复次数计算实际冻结时长（秒）。"""

        base = max(1, int(self.seconds))
        if not self.escalate or repeat <= 1:
            return base
        duration = base * (2 ** min(int(repeat) - 1, 10))
        return min(duration, max(base, int(self.max_seconds)))


def _get(spec: Any, name: str) -> Any:
    """从对象或字典里取字段。"""

    if isinstance(spec, dict):
        return spec.get(name)
    return getattr(spec, name, None)


def _coerce_escalate(value: Any) -> Optional[bool]:
    """把配置值转成三态：``True`` / ``False`` / ``None``（跟随全局）。"""

    text = str(value if value is not None else ESCALATE_INHERIT).strip().lower()
    if text in {"on", "true", "1", "yes", "开", "开启"}:
        return True
    if text in {"off", "false", "0", "no", "关", "关闭"}:
        return False
    return None


def build_rules(specs: Iterable[Any]) -> Tuple[List[FreezeRule], List[str]]:
    """把配置项解析成规则列表。

    Args:
        specs: 配置页里的规则项（具备 ``model`` / ``task`` / ``duration`` 等字段的对象或字典）。

    Returns:
        tuple[list[FreezeRule], list[str]]: 有效规则与问题描述（用于日志告警）。
    """

    rules: List[FreezeRule] = []
    problems: List[str] = []
    for index, spec in enumerate(specs or []):
        model_pattern = str(_get(spec, "model") or "").strip()
        task = str(_get(spec, "task") or "").strip()
        raw_duration = _get(spec, "duration")
        seconds = parse_duration(raw_duration, 0)
        if not model_pattern and not task:
            problems.append(f"规则 #{index + 1} 既没有填模型也没有填任务，已忽略")
            continue
        if seconds <= 0:
            problems.append(
                f"规则 #{index + 1}（{task or '*'} / {model_pattern or '*'}）"
                f"的冻结时长「{raw_duration}」无法识别（示例：30m / 5h / 1d / 1800），已忽略"
            )
            continue

        max_seconds = parse_duration(_get(spec, "max_duration"), 0)
        window_seconds = parse_duration(_get(spec, "window_duration"), 0)
        if max_seconds and max_seconds < seconds:
            problems.append(
                f"规则 #{index + 1} 的时长上限（{format_duration(max_seconds)}）小于冻结时长"
                f"（{format_duration(seconds)}），翻倍将不会生效"
            )
        rules.append(
            FreezeRule(
                index=index,
                model_pattern=model_pattern,
                task=task,
                seconds=seconds,
                escalate=_coerce_escalate(_get(spec, "escalate")),
                max_seconds=max_seconds or None,
                window_seconds=window_seconds or None,
                note=str(_get(spec, "note") or "").strip(),
            )
        )
    return rules, problems


def resolve_plan(
    task: str,
    model: str,
    rules: Sequence[FreezeRule],
    *,
    default_seconds: int,
    default_escalate: bool,
    default_max_seconds: int,
    default_window_seconds: int,
) -> FreezePlan:
    """解析出某个 (任务, 模型) 的冻结计划：第一条命中的规则优先，否则用全局默认。"""

    for rule in rules:
        if rule.matches(task, model):
            return FreezePlan(
                seconds=rule.seconds,
                escalate=default_escalate if rule.escalate is None else rule.escalate,
                max_seconds=rule.max_seconds if rule.max_seconds is not None else default_max_seconds,
                window_seconds=rule.window_seconds if rule.window_seconds is not None else default_window_seconds,
                rule=rule,
            )
    return FreezePlan(
        seconds=max(1, int(default_seconds)),
        escalate=bool(default_escalate),
        max_seconds=max(1, int(default_max_seconds)),
        window_seconds=max(0, int(default_window_seconds)),
    )


def unmatched_rules(rules: Sequence[FreezeRule], task_lists: dict) -> List[str]:
    """找出没有匹配到任何真实模型的规则（多半是通配符或任务名写错了）。

    Args:
        rules: 已解析的规则。
        task_lists: ``{任务: [模型...]}``。

    Returns:
        list[str]: 未命中规则的描述。
    """

    pairs = [
        (str(task), str(model))
        for task, models in (task_lists or {}).items()
        for model in (models or [])
    ]
    result: List[str] = []
    for rule in rules:
        if not any(rule.matches(task, model) for task, model in pairs):
            result.append(rule.describe())
    return result

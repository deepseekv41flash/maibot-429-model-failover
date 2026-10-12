"""插件配置快照。

把 SDK 的插件配置模型转换成不可变快照，冻结管理器只依赖快照，
因此可以脱离宿主在离线测试里直接构造。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence, Tuple

from .rules import FreezePlan, FreezeRule, build_rules, resolve_plan

DEFAULT_EXCLUDE_TASKS = ("embedding", "image_embedding")
"""默认排除的任务：嵌入类模型的向量空间必须保持一致，切换会污染向量库。"""


def _to_str_tuple(raw: Iterable[object]) -> Tuple[str, ...]:
    """把配置里的列表项规范化为去空白的字符串元组。"""

    result: List[str] = []
    for item in raw or ():
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class FailoverSettings:
    """一次决策所需的全部配置。"""

    enabled: bool = True
    admin_user_ids: Tuple[str, ...] = ()
    allow_llm_unfreeze: bool = False
    """是否允许模型通过 ``unfreeze_model`` 工具解冻（默认关闭，见 README 的权限边界说明）。"""
    poll_interval_seconds: float = 3.0
    status_codes: Tuple[int, ...] = (429,)
    detect_rate_limit_by_type: bool = True
    enable_snapshot_scan: bool = True
    enable_log_scan: bool = True
    scan_window_seconds: float = 30.0
    max_log_age_seconds: float = 300.0
    skip_if_no_valid_alternative: bool = True
    freeze_seconds: int = 180
    escalate: bool = True
    max_freeze_seconds: int = 1800
    escalation_window_seconds: int = 900
    model_rules: Tuple[FreezeRule, ...] = ()
    rule_problems: Tuple[str, ...] = ()
    freeze_scope: str = "task"
    exclude_tasks: Tuple[str, ...] = DEFAULT_EXCLUDE_TASKS
    exclude_models: Tuple[str, ...] = ()
    min_alternatives: int = 1
    dry_run: bool = False
    restore_on_unload: bool = True
    reassert_on_external_change: bool = True
    respect_manual_edit: bool = True
    notify_enabled: bool = False
    notify_stream_id: str = ""
    notify_on_freeze: bool = True
    notify_on_unfreeze: bool = True
    maibot_root: str = ""
    model_config_path: str = ""
    log_dir: str = ""
    backup_keep: int = 10

    @property
    def global_scope(self) -> bool:
        """冻结范围是否为“该模型在所有任务里都移除”。"""

        return str(self.freeze_scope).strip().lower() == "model"

    def freeze_duration(self, repeat: int) -> int:
        """按重复次数计算本次冻结时长（秒）——不含按模型规则的全局默认值。"""

        base = max(1, int(self.freeze_seconds))
        if not self.escalate or repeat <= 1:
            return base
        duration = base * (2 ** min(repeat - 1, 10))
        return min(duration, max(base, int(self.max_freeze_seconds)))

    def plan_for(self, task: str, model: str) -> FreezePlan:
        """解析指定 (任务, 模型) 的冻结计划：先匹配「按模型规则」，否则用全局默认。"""

        return resolve_plan(
            task,
            model,
            self.model_rules,
            default_seconds=self.freeze_seconds,
            default_escalate=self.escalate,
            default_max_seconds=self.max_freeze_seconds,
            default_window_seconds=self.escalation_window_seconds,
        )

    def freeze_duration_for(self, task: str, model: str, repeat: int) -> int:
        """按 (任务, 模型) 规则 + 重复次数计算冻结时长（秒）。"""

        return self.plan_for(task, model).duration(repeat)

    def escalation_window_for(self, task: str, model: str) -> int:
        """按 (任务, 模型) 规则返回「重复触发」判定窗口（秒）。"""

        return self.plan_for(task, model).window_seconds

    def rule_labels(self) -> Tuple[str, ...]:
        """全部规则的描述（用于日志与状态展示）。"""

        return tuple(rule.describe() for rule in self.model_rules)

    def plan_lines(self) -> Tuple[str, ...]:
        """状态页展示用的规则摘要。"""

        if not self.model_rules:
            return ()
        lines = ["按模型规则（先匹配先赢）："]
        lines.extend(f"· {rule.describe()}" for rule in self.model_rules)
        lines.extend(f"⚠ {problem}" for problem in self.rule_problems)
        return tuple(lines)

    def is_task_excluded(self, task: str) -> bool:
        """判断任务（模型类别）是否被排除。"""

        return task in self.exclude_tasks

    def is_model_excluded(self, model: str) -> bool:
        """判断模型是否被排除（支持 ``fnmatch`` 通配）。"""

        if not self.exclude_models:
            return False
        from fnmatch import fnmatch

        return any(fnmatch(model, pattern) for pattern in self.exclude_models)

    def is_admin(self, user_id: str, is_local_operator: bool = False) -> bool:
        """判断调用者是否为插件管理员。"""

        if is_local_operator:
            return True
        if not self.admin_user_ids:
            return False
        return str(user_id).strip() in self.admin_user_ids


def build_settings(config: object) -> FailoverSettings:
    """从 SDK 配置对象构造快照。

    Args:
        config: ``PluginConfigBase`` 实例（或任意具备同名属性的对象）。

    Returns:
        FailoverSettings: 不可变配置快照。
    """

    plugin = getattr(config, "plugin", None)
    detect = getattr(config, "detect", None)
    freeze = getattr(config, "freeze", None)
    notify = getattr(config, "notify", None)
    advanced = getattr(config, "advanced", None)

    def pick(section: object, name: str, default: object) -> object:
        """读取配置节字段，缺字段时回退默认值。"""

        value = getattr(section, name, None)
        return default if value is None else value

    def pick_codes(section: object) -> Tuple[int, ...]:
        """读取状态码列表。"""

        raw = pick(section, "trigger_status_codes", [429])
        codes: List[int] = []
        for item in raw if isinstance(raw, Sequence) else [raw]:
            try:
                code = int(str(item).strip())
            except (TypeError, ValueError):
                continue
            if 100 <= code <= 599 and code not in codes:
                codes.append(code)
        return tuple(codes) or (429,)

    raw_rules = pick(freeze, "model_rules", []) or []
    model_rules, rule_problems = build_rules(raw_rules if isinstance(raw_rules, Sequence) else [])

    return FailoverSettings(
        enabled=bool(pick(plugin, "enabled", True)),
        admin_user_ids=_to_str_tuple(pick(plugin, "admin_user_ids", [])),  # type: ignore[arg-type]
        allow_llm_unfreeze=bool(pick(plugin, "allow_llm_unfreeze", False)),
        poll_interval_seconds=max(0.5, float(pick(detect, "poll_interval_seconds", 3.0))),
        status_codes=pick_codes(detect),
        detect_rate_limit_by_type=bool(pick(detect, "detect_rate_limit_by_type", True)),
        enable_snapshot_scan=bool(pick(detect, "enable_snapshot_scan", True)),
        enable_log_scan=bool(pick(detect, "enable_log_scan", True)),
        scan_window_seconds=max(1.0, float(pick(detect, "scan_window_seconds", 30.0))),
        max_log_age_seconds=max(0.0, float(pick(detect, "max_log_age_seconds", 300.0))),
        skip_if_no_valid_alternative=bool(pick(freeze, "skip_if_no_valid_alternative", True)),
        freeze_seconds=max(1, int(pick(freeze, "freeze_seconds", 180))),
        escalate=bool(pick(freeze, "escalate", True)),
        max_freeze_seconds=max(1, int(pick(freeze, "max_freeze_seconds", 1800))),
        escalation_window_seconds=max(0, int(pick(freeze, "escalation_window_seconds", 900))),
        model_rules=tuple(model_rules),
        rule_problems=tuple(rule_problems),
        freeze_scope=str(pick(freeze, "freeze_scope", "task") or "task").strip().lower(),
        exclude_tasks=_to_str_tuple(pick(freeze, "exclude_tasks", DEFAULT_EXCLUDE_TASKS)),  # type: ignore[arg-type]
        exclude_models=_to_str_tuple(pick(freeze, "exclude_models", [])),  # type: ignore[arg-type]
        min_alternatives=max(0, int(pick(freeze, "min_alternatives", 1))),
        dry_run=bool(pick(freeze, "dry_run", False)),
        restore_on_unload=bool(pick(freeze, "restore_on_unload", True)),
        reassert_on_external_change=bool(pick(freeze, "reassert_on_external_change", True)),
        respect_manual_edit=bool(pick(freeze, "respect_manual_edit", True)),
        notify_enabled=bool(pick(notify, "enabled", False)),
        notify_stream_id=str(pick(notify, "stream_id", "") or "").strip(),
        notify_on_freeze=bool(pick(notify, "notify_on_freeze", True)),
        notify_on_unfreeze=bool(pick(notify, "notify_on_unfreeze", True)),
        maibot_root=str(pick(advanced, "maibot_root", "") or "").strip(),
        model_config_path=str(pick(advanced, "model_config_path", "") or "").strip(),
        log_dir=str(pick(advanced, "log_dir", "") or "").strip(),
        backup_keep=max(0, int(pick(advanced, "backup_keep", 10))),
    )

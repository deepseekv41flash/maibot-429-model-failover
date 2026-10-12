"""429 限流熔断插件的内部实现包。

模块划分：
    host_paths   —— 宿主机根目录 / 模型配置 / 日志目录的探测
    host_config  —— 宿主 model_config.toml 的安全读写（外科式改写 + 原子替换）
    detector     —— 从失败快照与应用日志中识别 429 等限流事件
    state        —— 冻结记录与统计的持久化
    settings     —— 插件配置快照
    freeze       —— 冻结/还原调度器
"""

from .detector import DetectedRateLimit, FailureDetector, normalize_status_codes
from .freeze import FreezeManager
from .host_config import ApplyResult, HostModelConfig, ModelConfigError
from .settings import DEFAULT_EXCLUDE_TASKS, FailoverSettings, build_settings
from .state import FreezeEntry, PluginState, StateStore

__all__ = [
    "ApplyResult",
    "DEFAULT_EXCLUDE_TASKS",
    "DetectedRateLimit",
    "FailoverSettings",
    "FailureDetector",
    "FreezeEntry",
    "FreezeManager",
    "HostModelConfig",
    "ModelConfigError",
    "PluginState",
    "StateStore",
    "build_settings",
    "normalize_status_codes",
]

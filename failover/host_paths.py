"""定位 MaiBot 宿主根目录与插件需要读写的宿主文件路径。

插件进程由宿主 Runner 以子进程方式拉起，共享同一份容器文件系统，因此可以
读取宿主的失败快照/日志，也可以在宿主允许的范围内改写模型配置。本模块只做
“探测 + 校验”，不硬编码任何绝对路径：

1. 插件自身文件位置（``plugins/<某目录>/failover/host_paths.py``）向上回溯；
2. Runner 注入的环境变量 ``MAIBOT_PLUGIN_DIRS`` / ``MAIBOT_TRUSTED_PLUGIN_DIRS``；
3. 进程工作目录 ``PWD`` / ``os.getcwd()``；
4. 用户显式配置的根目录（优先级最高）。

候选目录只有在包含 ``config/model_config.toml`` 时才被认定为宿主根目录。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import os

MODEL_CONFIG_RELATIVE = Path("config") / "model_config.toml"
"""宿主模型配置相对根目录的位置。"""

LOG_DIR_RELATIVE = Path("logs")
"""宿主日志目录相对根目录的位置。"""

LLM_ERROR_DIR_RELATIVE = Path("logs") / "maisaka_prompt" / "llm_error"
"""宿主 LLM 失败快照目录相对根目录的位置。"""

_ENV_CANDIDATES = (
    "MAIBOT_PLUGIN_DIRS",
    "MAIBOT_TRUSTED_PLUGIN_DIRS",
    "PWD",
)

_MAX_PARENT_LEVELS = 6


def _iter_candidates(explicit_root: str = "") -> Iterable[Path]:
    """依次产出可能的宿主根目录候选。

    Args:
        explicit_root: 用户在插件配置里显式填写的宿主根目录。

    Yields:
        Path: 待校验的候选目录。
    """

    raw_explicit = str(explicit_root or "").strip()
    if raw_explicit:
        yield Path(raw_explicit).expanduser()

    for env_name in _ENV_CANDIDATES:
        raw_value = os.environ.get(env_name, "")
        if not raw_value:
            continue
        for part in raw_value.split(os.pathsep):
            if part.strip():
                yield Path(part.strip())

    try:
        yield Path(__file__).resolve().parents[3]
    except IndexError:  # pragma: no cover - 目录层级异常时忽略
        pass

    try:
        yield Path.cwd()
    except OSError:  # pragma: no cover - 工作目录不可用时忽略
        pass


def _looks_like_host_root(candidate: Path) -> bool:
    """判断候选目录是否包含宿主模型配置。"""

    try:
        return (candidate / MODEL_CONFIG_RELATIVE).is_file()
    except OSError:  # pragma: no cover - 权限异常时忽略
        return False


def resolve_host_root(explicit_root: str = "") -> Path | None:
    """解析 MaiBot 宿主根目录。

    Args:
        explicit_root: 用户在插件配置里显式填写的宿主根目录；为空时自动探测。

    Returns:
        Optional[Path]: 命中 ``config/model_config.toml`` 的宿主根目录；探测失败返回 ``None``。
    """

    seen: set[Path] = set()
    for candidate in _iter_candidates(explicit_root):
        try:
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover - 非法路径时忽略
            continue
        levels = [resolved, *list(resolved.parents)[:_MAX_PARENT_LEVELS]]
        for level in levels:
            if level in seen:
                continue
            seen.add(level)
            if _looks_like_host_root(level):
                return level
    return None


def resolve_model_config_path(explicit_root: str = "", explicit_path: str = "") -> Path | None:
    """解析宿主 ``model_config.toml`` 路径。

    Args:
        explicit_root: 显式宿主根目录。
        explicit_path: 显式模型配置文件路径（优先于根目录推导）。

    Returns:
        Optional[Path]: 模型配置文件路径；无法定位时返回 ``None``。
    """

    raw_path = str(explicit_path or "").strip()
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            root = resolve_host_root(explicit_root)
            if root is not None:
                candidate = root / candidate
        return candidate

    root = resolve_host_root(explicit_root)
    if root is None:
        return None
    return root / MODEL_CONFIG_RELATIVE


def resolve_log_dir(explicit_root: str = "", explicit_dir: str = "") -> Path | None:
    """解析宿主日志目录。

    Args:
        explicit_root: 显式宿主根目录。
        explicit_dir: 显式日志目录（优先于根目录推导）。

    Returns:
        Optional[Path]: 日志目录路径；无法定位时返回 ``None``。
    """

    raw_dir = str(explicit_dir or "").strip()
    if raw_dir:
        candidate = Path(raw_dir).expanduser()
        if not candidate.is_absolute():
            root = resolve_host_root(explicit_root)
            if root is not None:
                candidate = root / candidate
        return candidate

    root = resolve_host_root(explicit_root)
    if root is None:
        return None
    return root / LOG_DIR_RELATIVE


def resolve_llm_error_dir(explicit_root: str = "", explicit_dir: str = "") -> Path | None:
    """解析宿主 LLM 失败快照目录。

    Args:
        explicit_root: 显式宿主根目录。
        explicit_dir: 显式日志目录（快照目录位于其下）。

    Returns:
        Optional[Path]: 失败快照目录；无法定位时返回 ``None``。
    """

    log_dir = resolve_log_dir(explicit_root, explicit_dir)
    if log_dir is None:
        return None
    return log_dir / LLM_ERROR_DIR_RELATIVE.relative_to(LOG_DIR_RELATIVE)

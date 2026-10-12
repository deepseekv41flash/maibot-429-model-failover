"""宿主 ``model_config.toml`` 的安全读写。

MaiBot 的模型调度完全由 ``config/model_config.toml`` 里的
``[model_task_config.<任务>]`` → ``model_list`` 决定：``sequential`` 策略取列表
里第一个可用模型，``balance`` / ``random`` 也只会在这份列表里挑。

因此“冻结某个模型 + 切到下一个备选模型”的落点就是：把该模型从对应任务的
``model_list`` 里暂时移除，冻结到期再按原位置插回去。本模块负责这件事，并且：

* 只做外科式文本替换（数组字面量区间），文件其余部分逐字节保留；
* 写前用 ``tomllib`` 解析新旧文本并逐节比对，任何意外的差异都会中止写入；
* 先备份、再**就地写入**（保持 inode 不变），宿主 ``watchfiles`` 监听会捕获到
  ``Change.modified`` 并在约 1 秒内完成热重载；这里刻意不用
  「临时文件 + ``os.replace``」的原子替换写法，原因见 :meth:`HostModelConfig.write_in_place`；
* ``dry_run`` 模式下只产出结果，不落盘。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import contextlib
import os
import re
import shutil
import tomllib

TABLE_HEADER_RE = re.compile(r"^[ \t]*\[[^\]\r\n]+\][ \t]*(?:#.*)?$")
SECTION_HEADER_TEMPLATE = r"^[ \t]*\[[ \t]*model_task_config[ \t]*\.[ \t]*{name}[ \t]*\][ \t]*(?:#.*)?$"
MODEL_LIST_KEY_RE = re.compile(r"^[ \t]*model_list[ \t]*=[ \t]*", re.MULTILINE)
ITEM_INDENT_RE = re.compile(r"\n([ \t]+)\S")


class ModelConfigError(RuntimeError):
    """模型配置文件读写或校验失败。"""


@dataclass
class ApplyResult:
    """一次配置改写的结果。"""

    changed: bool
    reason: str
    new_text: str = ""
    backup_path: Optional[Path] = None
    written: bool = False


@dataclass(frozen=True)
class ArrayStyle:
    """``model_list`` 数组的排版风格，用于还原时保持原样。"""

    multiline: bool = True
    item_indent: str = "    "
    key_indent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可持久化的字典。"""

        return {"multiline": self.multiline, "item_indent": self.item_indent, "key_indent": self.key_indent}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> Optional["ArrayStyle"]:
        """从字典还原；非法输入返回 ``None``。"""

        if not isinstance(raw, Mapping):
            return None
        item_indent = raw.get("item_indent")
        key_indent = raw.get("key_indent")
        return cls(
            multiline=bool(raw.get("multiline", True)),
            item_indent=item_indent if isinstance(item_indent, str) else "    ",
            key_indent=key_indent if isinstance(key_indent, str) else "",
        )


@dataclass
class TaskTableRef:
    """某个任务表在文本中的位置信息。"""

    name: str
    header_start: int
    section_end: int


@dataclass
class ModelListRef:
    """某个任务表内 ``model_list`` 数组字面量的位置与风格。"""

    key_indent: str
    value_start: int
    value_end: int
    multiline: bool
    item_indent: str
    models: List[str] = field(default_factory=list)


def _toml_string(value: str) -> str:
    """把模型名渲染成 TOML 基础字符串字面量。"""

    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _scan_array_end(text: str, start: int) -> int:
    """从 ``[`` 开始扫描到配对的 ``]``，返回其后的下标。

    扫描时跳过基础字符串、字面字符串、多行字符串与注释，避免被内容里的括号干扰。

    Args:
        text: 完整文件文本。
        start: ``[`` 所在下标。

    Returns:
        int: 配对 ``]`` 的下一个字符下标。

    Raises:
        ModelConfigError: 数组未闭合时抛出。
    """

    if start >= len(text) or text[start] != "[":
        raise ModelConfigError("model_list 的值不是数组字面量，拒绝改写")

    index = start
    depth = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char == "#":
            newline = text.find("\n", index)
            index = length if newline < 0 else newline + 1
            continue

        if text.startswith('"""', index):
            end = text.find('"""', index + 3)
            index = length if end < 0 else end + 3
            continue

        if text.startswith("'''", index):
            end = text.find("'''", index + 3)
            index = length if end < 0 else end + 3
            continue

        if char == '"':
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == '"':
                    index += 1
                    break
                index += 1
            continue

        if char == "'":
            index += 1
            while index < length:
                if text[index] == "'":
                    index += 1
                    break
                index += 1
            continue

        if char == "[":
            depth += 1
            index += 1
            continue

        if char == "]":
            depth -= 1
            index += 1
            if depth == 0:
                return index
            continue

        index += 1

    raise ModelConfigError("model_list 数组没有闭合的 ]，文件可能已损坏")


class HostModelConfig:
    """宿主模型配置文件的安全编辑器。"""

    def __init__(self, path: Path, logger: Any) -> None:
        """初始化编辑器。

        Args:
            path: ``model_config.toml`` 路径。
            logger: 插件日志器。
        """

        self.path = Path(path)
        self._logger = logger

    # ------------------------------------------------------------------ 读取

    @property
    def exists(self) -> bool:
        """配置文件是否存在。"""

        return self.path.is_file()

    def read_text(self) -> str:
        """读取配置文件文本。

        Raises:
            ModelConfigError: 文件不存在或不可读时抛出。
        """

        if not self.exists:
            raise ModelConfigError(f"未找到宿主模型配置文件: {self.path}")
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ModelConfigError(f"读取宿主模型配置失败: {exc}") from exc

    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        """解析 TOML 文本。

        Raises:
            ModelConfigError: TOML 语法错误时抛出。
        """

        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ModelConfigError(f"TOML 解析失败: {exc}") from exc

    @staticmethod
    def task_tables(data: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        """取出 ``model_task_config`` 下的任务表。"""

        tasks = data.get("model_task_config")
        if not isinstance(tasks, dict):
            return {}
        return {str(name): value for name, value in tasks.items() if isinstance(value, dict)}

    @classmethod
    def task_model_lists(cls, data: Mapping[str, Any]) -> Dict[str, List[str]]:
        """取出每个任务的 ``model_list``。"""

        result: Dict[str, List[str]] = {}
        for name, table in cls.task_tables(data).items():
            raw_list = table.get("model_list")
            if isinstance(raw_list, list):
                result[name] = [str(item) for item in raw_list]
        return result

    @staticmethod
    def configured_model_names(data: Mapping[str, Any]) -> List[str]:
        """取出 ``[[models]]`` 中声明的全部模型名。"""

        models = data.get("models")
        if not isinstance(models, list):
            return []
        names: List[str] = []
        for entry in models:
            if isinstance(entry, dict):
                name = str(entry.get("name") or entry.get("model_identifier") or "").strip()
                if name:
                    names.append(name)
        return names

    @classmethod
    def undeclared_models(cls, data: Mapping[str, Any]) -> Dict[str, List[str]]:
        """找出任务列表里引用了、但没有在 ``[[models]]`` 中声明的模型。

        MaiBot 取模型时会走 ``get_model_info_by_name``，名字没声明就直接抛
        「未找到名为 X 的模型」，而且这条错误**不会被罚分**（发生在罚分之前），
        于是 ``balance`` 策略会一直挑中它、每次请求都失败。

        Returns:
            dict[str, list[str]]: ``{任务名: [未声明的模型名]}``。
        """

        declared = set(cls.configured_model_names(data))
        result: Dict[str, List[str]] = {}
        for task, models in cls.task_model_lists(data).items():
            missing = [name for name in models if name and name not in declared]
            if missing:
                result[task] = missing
        return result

    # ------------------------------------------------------------------ 定位

    def find_task_table(self, text: str, task: str) -> TaskTableRef:
        """定位某个任务表在文本中的区间。

        Raises:
            ModelConfigError: 找不到该任务表时抛出。
        """

        pattern = re.compile(SECTION_HEADER_TEMPLATE.format(name=re.escape(task)), re.MULTILINE)
        match = pattern.search(text)
        if match is None:
            raise ModelConfigError(f"模型配置里找不到任务表 [model_task_config.{task}]")

        header_start = match.start()
        section_end = len(text)
        search_from = match.end()
        while True:
            line_start = text.find("\n", search_from)
            if line_start < 0:
                break
            next_line_start = line_start + 1
            line_end = text.find("\n", next_line_start)
            candidate_line = text[next_line_start : line_end if line_end >= 0 else len(text)]
            if TABLE_HEADER_RE.match(candidate_line):
                section_end = next_line_start
                break
            search_from = next_line_start

        return TaskTableRef(name=task, header_start=header_start, section_end=section_end)

    def array_style(self, text: str, task: str) -> ArrayStyle:
        """读取某个任务 ``model_list`` 的排版风格。"""

        ref = self.find_model_list(text, task)
        return ArrayStyle(multiline=ref.multiline, item_indent=ref.item_indent, key_indent=ref.key_indent)

    def find_model_list(self, text: str, task: str) -> ModelListRef:
        """定位某个任务表内 ``model_list`` 数组字面量。

        Raises:
            ModelConfigError: 找不到 ``model_list`` 或其值不是数组时抛出。
        """

        table = self.find_task_table(text, task)
        section_text = text[table.header_start : table.section_end]
        match = MODEL_LIST_KEY_RE.search(section_text)
        if match is None:
            raise ModelConfigError(f"任务 [model_task_config.{task}] 内找不到 model_list 字段")

        key_indent = re.match(r"[ \t]*", match.group(0)).group(0)  # type: ignore[union-attr]
        value_start = table.header_start + match.end()
        if value_start >= len(text) or text[value_start] != "[":
            raise ModelConfigError(f"任务 [model_task_config.{task}] 的 model_list 不是数组，拒绝改写")

        value_end = _scan_array_end(text, value_start)
        array_text = text[value_start:value_end]
        multiline = "\n" in array_text
        indent_match = ITEM_INDENT_RE.search(array_text)
        item_indent = indent_match.group(1) if indent_match else f"{key_indent}    "

        return ModelListRef(
            key_indent=key_indent,
            value_start=value_start,
            value_end=value_end,
            multiline=multiline,
            item_indent=item_indent,
        )

    # ------------------------------------------------------------------ 渲染

    @staticmethod
    def render_array(
        models: Sequence[str],
        ref: ModelListRef,
        style: Optional[ArrayStyle] = None,
    ) -> str:
        """按原有（或显式指定的）排版风格渲染新的 ``model_list`` 数组字面量。

        Args:
            models: 新的模型名列表。
            ref: 从当前文件探测到的数组位置与风格。
            style: 显式风格；还原冻结时用它把排版恢复成冻结前的样子。
        """

        names = [str(model) for model in models]
        multiline = ref.multiline if style is None else style.multiline
        item_indent = ref.item_indent if style is None else style.item_indent
        key_indent = ref.key_indent if style is None else style.key_indent

        if not multiline:
            return "[" + ", ".join(_toml_string(name) for name in names) + "]"
        if not names:
            return "[]"
        lines = ["["]
        lines.extend(f"{item_indent}{_toml_string(name)}," for name in names)
        lines.append(f"{key_indent}]")
        return "\n".join(lines)

    def render(
        self,
        text: str,
        updates: Mapping[str, Sequence[str]],
        styles: Optional[Mapping[str, ArrayStyle]] = None,
    ) -> Tuple[str, Dict[str, List[str]]]:
        """按 ``{任务: 新模型列表}`` 渲染出新的文件文本。

        只替换目标任务的 ``model_list`` 数组区间，从后往前替换以免下标偏移。

        Returns:
            tuple[str, dict[str, list[str]]]: 新文本与本次真正生效的列表。
        """

        edits: List[Tuple[int, int, str]] = []
        applied: Dict[str, List[str]] = {}
        for task, models in updates.items():
            ref = self.find_model_list(text, task)
            new_array = self.render_array(models, ref, (styles or {}).get(task))
            if new_array == text[ref.value_start : ref.value_end]:
                continue
            edits.append((ref.value_start, ref.value_end, new_array))
            applied[task] = [str(model) for model in models]

        if not edits:
            return text, {}

        edits.sort(key=lambda item: item[0], reverse=True)
        new_text = text
        for start, end, replacement in edits:
            new_text = new_text[:start] + replacement + new_text[end:]
        return new_text, applied

    # ------------------------------------------------------------------ 校验

    def validate_change(
        self,
        old_text: str,
        new_text: str,
        applied: Mapping[str, Sequence[str]],
    ) -> None:
        """校验改写结果：除目标任务的 ``model_list`` 外，其余配置必须逐节一致。

        Raises:
            ModelConfigError: 出现非预期差异时抛出。
        """

        old_data = self.parse(old_text)
        new_data = self.parse(new_text)

        if set(old_data.keys()) != set(new_data.keys()):
            raise ModelConfigError("改写后顶层配置节点发生变化，已中止写入")

        for key in ("models", "api_providers", "inner"):
            if old_data.get(key) != new_data.get(key):
                raise ModelConfigError(f"改写后 {key} 配置发生变化，已中止写入")

        old_tasks = self.task_tables(old_data)
        new_tasks = self.task_tables(new_data)
        if set(old_tasks.keys()) != set(new_tasks.keys()):
            raise ModelConfigError("改写后任务表集合发生变化，已中止写入")

        for task, expected_list in applied.items():
            actual = new_tasks.get(task, {}).get("model_list")
            if [str(item) for item in expected_list] != [str(item) for item in (actual or [])]:
                raise ModelConfigError(f"任务 {task} 的 model_list 未按预期写入")

        for task, old_table in old_tasks.items():
            new_table = new_tasks.get(task, {})
            if task in applied:
                expected = dict(old_table)
                expected["model_list"] = [str(item) for item in applied[task]]
                if new_table != expected:
                    raise ModelConfigError(f"任务 {task} 出现 model_list 之外的改动，已中止写入")
                continue
            if new_table != old_table:
                raise ModelConfigError(f"任务 {task} 的内容被意外改动，已中止写入")

    # ------------------------------------------------------------------ 写入

    def backup(self, text: str, backup_dir: Path, keep: int = 10) -> Optional[Path]:
        """把当前文件文本备份到插件数据目录。

        备份是**完整**的宿主配置（含 ``[[api_providers]]`` 的 api_key），因此写完后
        收紧文件权限（POSIX 上 0600，只留属主可读写）；Windows 上 ``os.chmod`` 语义
        有限，失败时忽略，不影响备份本身。

        Args:
            text: 当前配置文本。
            backup_dir: 备份目录（插件数据目录下的 ``backups/``）。
            keep: 保留份数；``<= 0`` 表示不写备份（由调用方 ``apply()`` 提前跳过）。
        """

        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            target = backup_dir / f"model_config_{stamp}.toml"
            target.write_text(text, encoding="utf-8")
            with contextlib.suppress(OSError):
                os.chmod(target, 0o600)
            self._prune_backups(backup_dir, keep)
            return target
        except OSError as exc:
            self._logger.warning(f"备份宿主机型配置失败（不影响本次改写）: {exc}")
            return None

    @staticmethod
    def _prune_backups(backup_dir: Path, keep: int) -> None:
        """只保留最近 N 份备份。"""

        if keep <= 0:
            return
        try:
            backups = sorted(
                (item for item in backup_dir.glob("model_config_*.toml") if item.is_file()),
                key=lambda item: item.name,
            )
        except OSError:
            return
        for stale in backups[:-keep]:
            with contextlib.suppress(OSError):
                stale.unlink()

    def write_in_place(self, text: str) -> None:
        """**就地**写入配置文件（保持 inode 不变）。

        为什么不用「临时文件 + ``os.replace``」这种常规原子替换？因为宿主用
        ``watchfiles`` 监听 ``model_config.toml``，而 inotify 的监听是挂在
        **inode** 上的：`os.replace` 会把监听对象换成被替换掉的旧 inode，之后
        该路径上的任何改动宿主都收不到（实测：替换后宿主既不再热重载，也不会
        恢复，因为没有任何异常触发 watchfiles 重启）。

        因此这里改为：一次 ``pwrite`` 覆盖内容 + ``ftruncate`` 收尾 + ``fsync``，
        inode 保持不变，宿主能稳定收到 ``Change.modified`` 并触发热重载。
        宿主自身读取配置文件只在收到（带 600ms 防抖的）变更事件之后发生，
        因此不存在读到半截文件的窗口。

        平台差异：Windows 的 CRT 在未指定 ``O_BINARY`` 时按**文本模式**打开文件，
        ``os.write`` 会把每个 ``\\n`` 写成 ``\\r\\n``，紧接着的
        ``ftruncate(len(data))`` 就会把文件尾部截掉（宿主配置被写坏）。因此下面
        显式带上 ``O_BINARY``；POSIX 没有这个常量，``getattr`` 兜底为 0，
        在 Linux/macOS 上的行为与不带该标志完全一致。

        Raises:
            ModelConfigError: 写入失败时抛出。
        """

        data = text.encode("utf-8")
        flags = os.O_WRONLY | getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(str(self.path), flags)
        except OSError as exc:
            raise ModelConfigError(f"打开宿主模型配置失败: {exc}") from exc

        try:
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:  # pragma: no cover - 极少见的写入异常
                    raise OSError("写入返回 0 字节")
                offset += written
            os.ftruncate(descriptor, len(data))
            os.fsync(descriptor)
        except OSError as exc:
            raise ModelConfigError(f"写入宿主模型配置失败: {exc}") from exc
        finally:
            with contextlib.suppress(OSError):
                os.close(descriptor)

    def apply(
        self,
        updates: Mapping[str, Sequence[str]],
        *,
        styles: Optional[Mapping[str, ArrayStyle]] = None,
        backup_dir: Optional[Path] = None,
        backup_keep: int = 10,
        dry_run: bool = False,
    ) -> ApplyResult:
        """执行一次改写：读取 → 渲染 → 校验 → 备份 → 就地写入 → 复核。

        Args:
            updates: ``{任务: 新的模型列表}``。
            backup_dir: 备份目录；为 ``None`` 时跳过备份。
            backup_keep: 备份保留份数；``<= 0`` 表示**不写备份**（与配置项
                「保留备份份数」的字面语义一致；已有备份不会被删除）。
            dry_run: 为 ``True`` 时只渲染与校验，不写盘。

        Returns:
            ApplyResult: 本次改写结果。
        """

        if not updates:
            return ApplyResult(changed=False, reason="无需改写")

        old_text = self.read_text()
        new_text, applied = self.render(old_text, updates, styles)
        if not applied:
            return ApplyResult(changed=False, reason="目标列表与现状一致")

        self.validate_change(old_text, new_text, applied)

        if dry_run:
            return ApplyResult(changed=True, reason="演练模式：未写入", new_text=new_text, written=False)

        write_backup = backup_dir is not None and backup_keep > 0
        backup_path = self.backup(old_text, backup_dir, backup_keep) if write_backup else None
        self.write_in_place(new_text)

        try:
            confirmed_text = self.read_text()
        except ModelConfigError as exc:  # pragma: no cover - 写入后立刻读取失败极少见
            raise ModelConfigError(f"写入后复核失败: {exc}") from exc
        if confirmed_text != new_text:
            raise ModelConfigError("写入后复核发现文件内容与预期不一致")

        return ApplyResult(
            changed=True,
            reason="已写入并校验通过",
            new_text=new_text,
            backup_path=backup_path,
            written=True,
        )

    def restore_backup(self, backup_path: Path) -> None:
        """用备份文件覆盖当前配置（排障用）。"""

        try:
            shutil.copyfile(backup_path, self.path)
        except OSError as exc:
            raise ModelConfigError(f"从备份恢复失败: {exc}") from exc

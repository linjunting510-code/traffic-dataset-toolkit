# -*- coding: utf-8 -*-
"""类别定义加载与解析。

唯一权威定义在仓库根目录的 ``configs/classes.yaml``:

- ``classes``: id -> 规范名称(0-8 共 9 类)
- ``aliases``: 别名 -> 规范名称(匹配时大小写不敏感)

优先使用 PyYAML 解析;如果运行环境没有安装 PyYAML,
自动退回到内置的极简解析器(只支持本文件用到的简单结构:
一层嵌套的 ``key: value`` 映射 + ``#`` 注释)。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml  # type: ignore

    _HAS_YAML = True
except ImportError:  # pragma: no cover - 取决于运行环境
    _HAS_YAML = False


def default_config_path() -> Path:
    """返回默认的 classes.yaml 路径(仓库根目录 configs/ 下)。

    解析顺序:
    1. 环境变量 ``TDS_CLASSES_YAML``(若设置)
    2. 仓库根目录 ``configs/classes.yaml``
       (按 src 布局,本文件位于 ``src/traffic_dataset/classes.py``,
       上三级即仓库根目录)
    """
    env = os.environ.get("TDS_CLASSES_YAML")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs" / "classes.yaml"


def _parse_simple_yaml(text: str) -> Dict[str, Dict[str, str]]:
    """极简 YAML 子集解析器(PyYAML 缺失时的兜底)。

    只支持 classes.yaml 用到的结构::

        classes:
          0: Car
        aliases:
          motorbike: Motorcycle

    即:顶层的节(无缩进 ``key:``)+ 节内一层缩进的 ``key: value``。
    支持 ``#`` 注释、空行、键值两端的引号。不处理列表/多行/锚点等。
    """
    result: Dict[str, Dict[str, str]] = {}
    current_section: Optional[str] = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        # 去掉行内注释(本配置文件不会出现值里带 '#' 的情况)
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line[0] in (" ", "\t"):
            # 顶层节,如 "classes:"
            key = line.split(":", 1)[0].strip().strip("'\"")
            current_section = key
            result[current_section] = {}
        else:
            if current_section is None:
                raise ValueError(
                    f"classes.yaml 第 {lineno} 行:缩进内容出现在任何节之前"
                )
            k, sep, v = line.strip().partition(":")
            if not sep:
                raise ValueError(f"classes.yaml 第 {lineno} 行:缺少 ':' 分隔符")
            result[current_section][k.strip().strip("'\"")] = v.strip().strip("'\"")
    return result


class ClassRegistry:
    """类别注册表:提供 名称/别名 -> class_id 的解析(大小写不敏感)。"""

    def __init__(self, id_to_name: Dict[int, str], aliases: Dict[str, str]):
        # id -> 规范名称
        self.id_to_name: Dict[int, str] = dict(sorted(id_to_name.items()))
        # 别名(小写) -> 规范名称
        self.aliases: Dict[str, str] = {str(k).lower(): str(v) for k, v in aliases.items()}
        # 名称/别名(统一小写) -> class_id
        self._lookup: Dict[str, int] = {}
        for cid, name in self.id_to_name.items():
            self._lookup[name.lower()] = cid
        for alias_lower, canonical in self.aliases.items():
            cid = self._lookup.get(canonical.lower())
            if cid is None:
                # 别名指向的规范名不在 classes 里,配置写错了,直接报错更省心
                raise ValueError(
                    f"classes.yaml 中别名 {alias_lower!r} 指向的规范名 "
                    f"{canonical!r} 不在 classes 定义里"
                )
            self._lookup[alias_lower] = cid

    def name_to_id(self, name: object) -> Optional[int]:
        """把类别名/别名解析为 class_id;无法识别时返回 None。"""
        if name is None:
            return None
        return self._lookup.get(str(name).strip().lower())

    def id_to_name_str(self, class_id: int) -> str:
        """class_id -> 规范名称;未知 id 返回 'unknown'。"""
        return self.id_to_name.get(class_id, "unknown")

    @property
    def num_classes(self) -> int:
        return len(self.id_to_name)

    @property
    def names(self) -> List[str]:
        """按 id 顺序返回规范名称列表(可直接写进 YOLO classes.txt)。"""
        return [self.id_to_name[i] for i in sorted(self.id_to_name)]


def load_classes(path: Optional[os.PathLike | str] = None) -> ClassRegistry:
    """加载 classes.yaml,返回 ClassRegistry。

    :param path: 自定义 classes.yaml 路径;为 None 时用 :func:`default_config_path`。
    :raises FileNotFoundError: 配置文件不存在。
    :raises ValueError: 配置结构不符合预期。
    """
    cfg_path = Path(path) if path else default_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"找不到类别定义文件: {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")
    if _HAS_YAML:
        data = yaml.safe_load(text)
    else:
        data = _parse_simple_yaml(text)

    if not isinstance(data, dict) or "classes" not in data:
        raise ValueError(f"{cfg_path} 结构不正确:缺少顶层 'classes' 节")

    raw_classes = data["classes"] or {}
    raw_aliases = data.get("aliases") or {}
    if not isinstance(raw_classes, dict) or not isinstance(raw_aliases, dict):
        raise ValueError(f"{cfg_path} 结构不正确:'classes'/'aliases' 必须是映射")

    try:
        id_to_name = {int(k): str(v) for k, v in raw_classes.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{cfg_path} 中 classes 的 id 必须是整数: {exc}") from exc

    if not id_to_name:
        raise ValueError(f"{cfg_path} 中 classes 为空")

    return ClassRegistry(id_to_name, raw_aliases)

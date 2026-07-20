# -*- coding: utf-8 -*-
"""数据集校验:对一个事件目录(images/ + labels/)做体检。

检查项:
1. 图片-标注配对:有图无 txt 记警告;有 txt 无图(孤儿标注)记错误
2. labels/classes.txt 与 configs/classes.yaml 的 9 类逐行对比
3. 每行标注格式:5 列、class_id 在 0..N-1、坐标/宽高在 (0,1] 区间
4. 统计:图片数、标注文件数、每类框数分布、零样本类别提醒

退出码:存在 error 返回 1;只有 warning(或全部通过)返回 0。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .classes import ClassRegistry
from .rename import IMAGE_EXTS

# 报告中逐条列出的问题上限,超过后只显示总数,避免刷屏
_MAX_DETAIL_LINES = 20

# labels/ 下不参与图片-标注配对的元文件
_META_LABEL_FILES = {"classes.txt", "autolabel_report.txt"}


class _Report:
    """收集 error / warning,并在最后汇总。"""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _print_items(title: str, items: List[str], icon: str) -> None:
    """逐条打印问题列表,超过上限时折叠。"""
    print(f"  {icon} {title}(共 {len(items)} 个):")
    for item in items[:_MAX_DETAIL_LINES]:
        print(f"     {item}")
    if len(items) > _MAX_DETAIL_LINES:
        print(f"     ... 其余 {len(items) - _MAX_DETAIL_LINES} 个从略")


def _check_pairing(images_dir: Path, labels_dir: Path,
                   report: _Report) -> Tuple[List[str], List[str]]:
    """检查图片与标注的配对情况,返回 (图片 stem 列表, 标注 stem 列表)。"""
    image_stems = sorted(p.stem for p in images_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    label_stems = sorted(p.stem for p in labels_dir.glob("*.txt")
                         if p.name not in _META_LABEL_FILES)

    image_set, label_set = set(image_stems), set(label_stems)
    missing_txt = sorted(image_set - label_set)   # 有图无标注
    orphan_txt = sorted(label_set - image_set)    # 有标注无图(孤儿)

    print(f"  图片: {len(image_stems)} 张 (images/)")
    print(f"  标注: {len(label_stems)} 个 (labels/*.txt,"
          f"不含 classes.txt / autolabel_report.txt)")

    if missing_txt:
        for stem in missing_txt:
            report.warning(f"有图无标注: {stem}")
        _print_items("有图无标注(warning)", missing_txt, "⚠️")
    if orphan_txt:
        for stem in orphan_txt:
            report.error(f"孤儿标注(有 txt 无图): {stem}.txt")
        _print_items("孤儿标注(有 txt 无图,error)",
                     [f"{s}.txt" for s in orphan_txt], "❌")
    if not missing_txt and not orphan_txt:
        print("  ✅ 图片与标注一一配对")
    return image_stems, label_stems


def _check_classes_txt(labels_dir: Path, registry: ClassRegistry,
                       report: _Report) -> None:
    """对比 labels/classes.txt 与 configs/classes.yaml 的类别定义。"""
    classes_txt = labels_dir / "classes.txt"
    expected = registry.names  # 按 id 顺序的规范名
    if not classes_txt.is_file():
        print("  ℹ️ labels/classes.txt 不存在,跳过一致性检查")
        print("     (可用 tds convert 生成,或手动按 classes.yaml 创建)")
        return

    actual = [line.strip() for line in
              classes_txt.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if actual == expected:
        print(f"  ✅ classes.txt 与 configs/classes.yaml 一致({len(expected)} 类)")
        return

    report.error(
        f"classes.txt 与 configs/classes.yaml 不一致"
        f"(期望 {len(expected)} 行,实际 {len(actual)} 行)"
    )
    print(f"  ❌ classes.txt 与 configs/classes.yaml 不一致:"
          f"期望 {len(expected)} 行,实际 {len(actual)} 行")
    print(f"     {'行号':<4} {'期望(classes.yaml)':<22} 实际(classes.txt)")
    for i in range(max(len(expected), len(actual))):
        exp = expected[i] if i < len(expected) else "(缺行)"
        act = actual[i] if i < len(actual) else "(缺行)"
        mark = "  " if exp == act else "❌"
        print(f"   {mark} {i + 1:<4} {exp:<22} {act}")


def _check_label_format(labels_dir: Path, label_stems: List[str],
                        registry: ClassRegistry,
                        report: _Report) -> Dict[int, int]:
    """逐行检查标注格式,返回每类框数统计 {class_id: 数量}。"""
    per_class: Dict[int, int] = {cid: 0 for cid in registry.id_to_name}
    bad_lines: List[str] = []
    total_lines = 0

    for stem in label_stems:
        txt_path = labels_dir / f"{stem}.txt"
        try:
            content = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            bad_lines.append(f"{txt_path.name}: 文件不是 UTF-8 编码")
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue  # 空行不算问题
            total_lines += 1
            parts = line.split()
            where = f"{txt_path.name}:{lineno}"
            if len(parts) != 5:
                bad_lines.append(f"{where} — 应为 5 列,实际 {len(parts)} 列")
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                bad_lines.append(f"{where} — class_id 不是整数: {parts[0]!r}")
                continue
            if cid not in registry.id_to_name:
                bad_lines.append(
                    f"{where} — class_id {cid} 超出范围 "
                    f"0-{registry.num_classes - 1}"
                )
                continue
            try:
                values = [float(x) for x in parts[1:]]
            except ValueError:
                bad_lines.append(f"{where} — 坐标/宽高不是数字")
                continue
            bad_vals = [v for v in values if not (0.0 < v <= 1.0)]
            if bad_vals:
                bad_lines.append(
                    f"{where} — 坐标/宽高必须在 (0,1] 区间,"
                    f"越界值: {bad_vals}"
                )
                continue
            per_class[cid] += 1

    if bad_lines:
        for msg in bad_lines:
            report.error(f"标注格式错误: {msg}")
        _print_items("标注格式错误(error)", bad_lines, "❌")
    else:
        print(f"  ✅ 共检查 {total_lines} 行标注,全部合法")
    return per_class


def validate_event(event_dir: str | os.PathLike, registry: ClassRegistry) -> int:
    """校验一个事件目录,打印中文报告。返回退出码(有 error 为 1,否则 0)。"""
    event_dir = Path(event_dir)
    images_dir = event_dir / "images"
    labels_dir = event_dir / "labels"
    report = _Report()

    print("=" * 60)
    print(f"数据集校验报告: {event_dir}")
    print("=" * 60)

    if not event_dir.is_dir():
        print(f"\n❌ 目录不存在: {event_dir}")
        return 1
    if not images_dir.is_dir():
        print(f"\n❌ 缺少 images/ 子目录: {images_dir}")
        return 1

    # ---- 1. 图片-标注配对 ----
    print("\n[1] 图片-标注配对")
    if not labels_dir.is_dir():
        report.warning(f"缺少 labels/ 子目录: {labels_dir}(全部图片未标注)")
        print(f"  ⚠️ 缺少 labels/ 子目录,跳过配对/格式检查")
        image_stems = sorted(p.stem for p in images_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        label_stems: List[str] = []
        per_class: Dict[int, int] = {cid: 0 for cid in registry.id_to_name}
    else:
        image_stems, label_stems = _check_pairing(images_dir, labels_dir, report)

        # ---- 2. classes.txt 一致性 ----
        print("\n[2] classes.txt 一致性")
        _check_classes_txt(labels_dir, registry, report)

        # ---- 3. 标注格式 ----
        print("\n[3] 标注格式")
        per_class = _check_label_format(labels_dir, label_stems, registry, report)

    # ---- 4. 统计 ----
    print("\n[4] 统计")
    total_boxes = sum(per_class.values())
    print(f"  图片数:     {len(image_stems)}")
    print(f"  标注文件数: {len(label_stems)}")
    print(f"  边界框总数: {total_boxes}")
    if total_boxes > 0:
        print("  各类分布:")
        for cid in sorted(registry.id_to_name):
            count = per_class.get(cid, 0)
            pct = count / total_boxes * 100
            print(f"    {cid} {registry.id_to_name_str(cid):<14} "
                  f"{count:>6} 个 ({pct:5.1f}%)")
    zero_classes = [registry.id_to_name_str(cid)
                    for cid in sorted(registry.id_to_name)
                    if per_class.get(cid, 0) == 0]
    if zero_classes and total_boxes > 0:
        report.warning("零样本类别: " + ", ".join(zero_classes))
        print(f"\n  ⚠️ 零样本类别(一个框都没有,注意补充数据):")
        for name in zero_classes:
            print(f"     - {name}")

    # ---- 汇总 ----
    n_err, n_warn = len(report.errors), len(report.warnings)
    print("\n" + "=" * 60)
    if n_err:
        print(f"结果: {n_err} 个错误,{n_warn} 个警告 → ❌ 未通过,请先修复错误")
    elif n_warn:
        print(f"结果: 0 个错误,{n_warn} 个警告 → ✅ 通过(仅警告)")
    else:
        print("结果: 0 个错误,0 个警告 → ✅ 全部通过")
    print("=" * 60)
    return 1 if n_err else 0

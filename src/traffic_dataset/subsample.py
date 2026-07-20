# -*- coding: utf-8 -*-
"""帧抽稀:固定监控机位下相邻帧近似重复,正式数据集按"每 N 帧标 1 帧"
抽稀(例子数据集 001537 即每 5 帧标 1 帧),先把图片抽稀再标注,
直接减少人工量。

行为:

- 读取 ``<event_dir>/images/`` 的图片(按文件名排序,保帧序),
  每 ``--every`` 帧取 1 帧;``--offset`` 控制起始位移
  (0 = 第 0,5,10... 索引,即"从第 1 张开始每 5 帧取 1 帧")
- 选中的图片复制到 ``<new_event_dir>/images/``,**保留原文件名**
  (帧号可追溯回源视频,文本标注才能对上)
- 源有 ``labels/`` 时:同名 .txt 一并复制;没有对应 txt 的图片
  照常复制并打印"有图无标注"提示;classes.txt 若存在也复制;
  autolabel_report.txt 等元文件不复制
- 输出目录已存在且非空时报错退出 1(防覆盖),除非 ``--force``

设计约定:选帧索引计算(纯逻辑)与文件复制分离,便于测试。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from .rename import IMAGE_EXTS

# labels/ 下需要复制的元文件(autolabel_report.txt 是旧报告的产物,
# 抽稀后会过时,不复制)
_META_TO_COPY = ("classes.txt",)


# ==================== 纯逻辑(不碰文件,便于单测) ====================

def select_indices(total: int, every: int, offset: int = 0) -> List[int]:
    """计算选中帧的索引:offset, offset+every, offset+2*every, ...

    >>> select_indices(120, 5, 0)  # 0,5,10,...,115 -> 24 帧
    """
    if every < 1:
        raise ValueError(f"--every 必须 >= 1,实际: {every}")
    if offset < 0:
        raise ValueError(f"--offset 必须 >= 0,实际: {offset}")
    if total < 0:
        raise ValueError(f"帧总数不能为负: {total}")
    return list(range(offset, total, every))


# ==================== 编排(文件复制) ====================

def subsample(event_dir: str | Path, out_dir: str | Path,
              every: int = 5, offset: int = 0, force: bool = False) -> int:
    """帧抽稀主流程。返回退出码(0 成功,1 失败)。"""
    event_dir = Path(event_dir)
    out_dir = Path(out_dir)
    images_dir = event_dir / "images"
    labels_dir = event_dir / "labels"

    # ---- 参数与输入校验 ----
    try:
        # 提前校验 every/offset(此处 total 无所谓,只为触发 ValueError)
        select_indices(0, every, offset)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1
    if not images_dir.is_dir():
        print(f"❌ 找不到 images 文件夹: {images_dir}")
        return 1
    images = sorted(p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"❌ images 里没有图片({'/'.join(IMAGE_EXTS)}): {images_dir}")
        return 1

    selected = [images[i] for i in select_indices(len(images), every, offset)]
    if not selected:
        print(f"❌ 选帧结果为空:共 {len(images)} 帧,"
              f"every={every},offset={offset}。请减小 --offset。")
        return 1

    # ---- 输出目录防覆盖 ----
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            print(f"❌ 输出目录已存在且非空: {out_dir}")
            print("   为避免覆盖已有数据,已中止。确认要覆盖请加 --force。")
            return 1
        print(f"⚠️ --force:将写入已存在的非空目录 {out_dir}(同名文件会被覆盖)")

    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    has_labels = labels_dir.is_dir()
    out_labels = out_dir / "labels"
    copied_labels = 0
    no_label: List[str] = []  # 有图无标注的图片名

    # ---- 复制图片(保留原文件名)与对应标注 ----
    print(f"源: {event_dir}")
    print(f"抽稀: 每 {every} 帧取 1 帧(offset={offset}),"
          f"{len(images)} 帧 -> {len(selected)} 帧\n")
    for img in selected:
        shutil.copy2(img, out_images / img.name)
        if has_labels:
            src_txt = labels_dir / f"{img.stem}.txt"
            if src_txt.is_file():
                out_labels.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_txt, out_labels / src_txt.name)
                copied_labels += 1
            else:
                no_label.append(img.name)

    # ---- 元文件:classes.txt 复制,autolabel_report.txt 不复制 ----
    if has_labels:
        for meta in _META_TO_COPY:
            src_meta = labels_dir / meta
            if src_meta.is_file():
                out_labels.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_meta, out_labels / meta)

    # ---- 提示与统计 ----
    if not has_labels:
        print("ℹ️ 源目录没有 labels/,只复制了图片(抽稀后再标注,正是推荐用法)")
    if no_label:
        print(f"⚠️ 有图无标注(共 {len(no_label)} 张,图片已照常复制):")
        for name in no_label[:10]:
            print(f"   {name}")
        if len(no_label) > 10:
            print(f"   ... 其余 {len(no_label) - 10} 张从略")

    print(f"\n{'=' * 50}")
    print(f"抽稀完成: {len(images)} 帧 -> {len(selected)} 帧"
          f"(比例 1/{every}),输出在 {out_dir}")
    if has_labels:
        print(f"复制标注: {copied_labels} 个 txt"
              f"{'(含 classes.txt)' if (out_labels / 'classes.txt').is_file() else ''}")
    print(f"帧号保留原文件名,可追溯回源视频。")
    print(f"{'=' * 50}")
    return 0

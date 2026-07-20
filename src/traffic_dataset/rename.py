# -*- coding: utf-8 -*-
"""图片批量重命名:把一个文件夹里的图片按文件名顺序重命名为
000001.jpg、000002.jpg ... 连续编号。

迁移自 scripts/rename_images.py,安全设计保持不变:

- 按文件名排序后顺序编号,保证连续帧顺序不乱
- 只处理图片(jpg/jpeg/png),不碰其他文件(如 classes.txt / json)
- 两步改名(先临时名、再最终名),避免新旧文件名互相覆盖
- 支持 --dry-run 预览,不动文件
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

# 只处理这些图片格式(小写后缀)
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def build_plan(folder: Path, start: int = 1, digits: int = 6) -> List[Tuple[str, str]]:
    """生成重命名计划:返回 [(旧文件名, 新文件名), ...](仅文件名,不含目录)。"""
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]
    files.sort()  # 按原文件名排序,连续帧通常名字里带递增编号
    plan = []
    for i, old in enumerate(files, start=start):
        # 统一用 .jpg 后缀(原名即使是 png 也只改名字,内容不变)
        plan.append((old, f"{i:0{digits}d}.jpg"))
    return plan


def rename_folder(folder: str | os.PathLike, dry_run: bool = False,
                  start: int = 1, digits: int = 6) -> int:
    """执行重命名。返回退出码(0 成功,1 失败)。"""
    folder = Path(folder)
    if not folder.is_dir():
        print(f"❌ 文件夹不存在: {folder}")
        return 1

    plan = build_plan(folder, start=start, digits=digits)
    if not plan:
        print(f"❌ 文件夹里没有图片({'/'.join(IMAGE_EXTS)}): {folder}")
        return 1

    print(f"找到 {len(plan)} 张图片,重命名计划预览:\n")
    for old, new in plan[:5]:
        print(f"  {old}  ->  {new}")
    if len(plan) > 7:
        print(f"  ... (共 {len(plan)} 个,中间省略)")
    for old, new in (plan[-2:] if len(plan) > 7 else plan[5:]):
        print(f"  {old}  ->  {new}")

    # 已经是目标名字的文件不会变(如 000001.jpg -> 000001.jpg),提示一下
    unchanged = sum(1 for old, new in plan if old == new)
    if unchanged:
        print(f"\n  注:其中 {unchanged} 个文件改名前后名字相同。")

    if dry_run:
        print("\n🔍 预览模式(--dry-run),没有真正改名。")
        print("   确认无误后去掉 --dry-run 再运行一次。")
        return 0

    # 真正改名:分两步,先全部改成临时名,避免覆盖冲突
    print("\n开始重命名...")
    tmp_map: List[Tuple[str, str]] = []
    for i, (old, _new) in enumerate(plan):
        tmp = f"__tmp_{i}__{Path(old).suffix.lower()}"
        os.rename(folder / old, folder / tmp)
        tmp_map.append((tmp, plan[i][1]))
    for tmp, new in tmp_map:
        os.rename(folder / tmp, folder / new)

    print(f"✅ 完成!{len(plan)} 张图片已重命名为 "
          f"{plan[0][1]} 起的连续编号。")
    return 0

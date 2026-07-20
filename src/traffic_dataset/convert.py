# -*- coding: utf-8 -*-
"""X-AnyLabeling JSON 标注 -> YOLO txt 转换。

迁移自 scripts/json_to_yolo.py:

- 读取 ``<event_dir>/images/*.json``
- 类别名/别名(大小写不敏感)映射为 class_id(定义见 configs/classes.yaml)
- 不在 9 类里的目标自动跳过并统计(清理误检)
- 输出到 ``<event_dir>/labels/*.txt``(与 json 同名)
- 额外写出 ``labels/classes.txt``(9 类规范名),让数据集自描述
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from .classes import ClassRegistry


def convert_event(event_dir: str | os.PathLike, registry: ClassRegistry) -> int:
    """转换一个事件目录。返回退出码(0 成功,1 失败)。"""
    event_dir = Path(event_dir)
    images_dir = event_dir / "images"
    labels_dir = event_dir / "labels"

    if not images_dir.is_dir():
        print(f"❌ 找不到 images 文件夹: {images_dir}")
        return 1
    labels_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(images_dir.glob("*.json"))
    if not json_files:
        print(f"❌ images 里没有 json 文件: {images_dir}")
        return 1

    print(f"找到 {len(json_files)} 个 json,开始转换...\n")
    total_boxes = 0
    skipped: Dict[str, int] = {}  # 被跳过的类别(不在 9 类里) -> 次数

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height:
            print(f"⚠️ {jf.name}: 缺少 imageWidth/imageHeight,已跳过")
            continue

        lines = []
        for shape in data.get("shapes", []):
            cid = registry.name_to_id(shape.get("label", ""))
            if cid is None:
                # 不在 9 类里的目标:跳过(自动清理误检)并统计
                key = str(shape.get("label", "")).strip().lower()
                skipped[key] = skipped.get(key, 0) + 1
                continue

            pts = shape.get("points", [])
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # 转成 YOLO 归一化格式:中心点 + 宽高
            xc = (x1 + x2) / 2 / width
            yc = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            total_boxes += 1

        # 写 txt(与 json 同名,放进 labels/)
        out = labels_dir / (jf.stem + ".txt")
        out.write_text("\n".join(lines), encoding="utf-8")

    # 顺手写出 classes.txt,让数据集自描述(与 configs/classes.yaml 一致)
    classes_txt = labels_dir / "classes.txt"
    classes_txt.write_text("\n".join(registry.names) + "\n", encoding="utf-8")

    print(f"✅ 完成!共写出 {len(json_files)} 个 txt,{total_boxes} 个框。")
    print(f"   位置: {labels_dir}")
    print(f"   已写出 classes.txt({registry.num_classes} 类规范名)")
    if skipped:
        print("\n⚠️ 以下类别不在 9 类里,已自动跳过(误检清理):")
        for name, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"   {name}: {count} 个")
    return 0

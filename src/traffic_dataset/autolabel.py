# -*- coding: utf-8 -*-
"""检测模型自动打标(ultralytics YOLO):对 ``<event_dir>/images/*.jpg``
跑 COCO 预训练模型,把能映射到项目 9 类的检出框写成 YOLO txt。

COCO → 项目 9 类映射(写死在这里,COCO 80 类里只有这 6 类对我们有用):

- person(0)        → 4 Pedestrian
- car(2)           → 0 Car
- motorcycle(3)    → 3 Motorcycle
- bus(5)           → 1 Bus
- truck(7)         → 2 Truck
- traffic light(9) → 8 Traffic light

其余 COCO 类(bicycle / potted plant 等)跳过并统计。
**Traffic Cone(5) / Barrier(6) / Tree(7) 三类 COCO 不覆盖,需人工补标**,
报告里会明确提醒。

工具包本体不硬依赖 ultralytics:这里惰性 import,缺失时打印中文
安装提示并返回退出码 1,不影响其他子命令。

设计约定:纯逻辑(COCO 映射、txt 行生成、置信度统计、报告生成)
与 ultralytics 调用分离,便于 mock 测试。
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .classes import ClassRegistry
from .rename import IMAGE_EXTS

# COCO class_id -> 项目 class_id(见模块 docstring 说明)
COCO_TO_PROJECT: Dict[int, int] = {
    0: 4,  # person        -> Pedestrian
    2: 0,  # car           -> Car
    3: 3,  # motorcycle    -> Motorcycle
    5: 1,  # bus           -> Bus
    7: 2,  # truck         -> Truck
    9: 8,  # traffic light -> Traffic light
}

# COCO 不覆盖、必须人工补标的项目类别 id
HUMAN_ONLY_IDS = (5, 6, 7)  # Traffic Cone / Barrier / Tree

# 低置信度帧清单长度(供人工优先复核)
LOW_CONF_TOP_N = 10

# 一个检出框:(coco_cls, conf, x_center, y_center, w, h),坐标已归一化
Box = Tuple[int, float, float, float, float, float]

# 一帧的统计:图片名、保留框数、平均置信度(零检出帧为 None)
FrameStat = Tuple[str, int, Optional[float]]

# 推理函数类型:输入图片路径,返回该图的检出框列表
InferFn = Callable[[Path], List[Box]]


# ==================== 纯逻辑(不碰 ultralytics,便于单测) ====================

def map_coco_class(coco_id: int) -> Optional[int]:
    """COCO class_id -> 项目 class_id;不在映射表里的返回 None。"""
    return COCO_TO_PROJECT.get(int(coco_id))


def yolo_line(class_id: int, x: float, y: float, w: float, h: float) -> str:
    """生成一行 YOLO 标注(6 位小数,与 convert.py 风格一致)。"""
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def boxes_to_lines(boxes: List[Box]) -> Tuple[List[str], List[float], Counter]:
    """把一帧的检出框转成 YOLO 行。

    返回 (行列表, 保留框的置信度列表, 被跳过的 COCO 类统计)。
    坐标做 [0,1] 截断(模型偶尔输出轻微越界值,避免 validate 报错)。
    """
    lines: List[str] = []
    confs: List[float] = []
    skipped: Counter = Counter()
    for coco_cls, conf, x, y, w, h in boxes:
        cid = map_coco_class(coco_cls)
        if cid is None:
            skipped[int(coco_cls)] += 1
            continue
        vals = [min(1.0, max(0.0, v)) for v in (x, y, w, h)]
        if vals[2] <= 0 or vals[3] <= 0:
            continue  # 宽高为 0 的退化框,丢弃
        lines.append(yolo_line(cid, *vals))
        confs.append(float(conf))
    return lines, confs, skipped


def mean_conf(confs: List[float]) -> Optional[float]:
    """平均置信度;空列表返回 None(零检出帧)。"""
    return sum(confs) / len(confs) if confs else None


def lowest_conf_frames(stats: List[FrameStat], n: int = LOW_CONF_TOP_N) -> List[FrameStat]:
    """每帧平均置信度最低的 n 帧(零检出帧不参与,单独列零检出清单)。"""
    with_boxes = [s for s in stats if s[2] is not None]
    return sorted(with_boxes, key=lambda s: (s[2], s[0]))[:n]


def zero_detection_frames(stats: List[FrameStat]) -> List[str]:
    """零检出帧(可能漏标,建议人工检查)的图片名列表。"""
    return sorted(name for name, count, _ in stats if count == 0)


def build_report(event_dir: Path, stats: List[FrameStat],
                 per_class: Dict[int, int], skipped: Counter,
                 registry: ClassRegistry, model_name: str, conf: float,
                 device: str, track: bool, elapsed: float,
                 coco_names: Optional[Dict[int, str]] = None) -> str:
    """生成抽检报告文本(打印 + 写 autolabel_report.txt 共用)。"""
    total_images = len(stats)
    total_boxes = sum(per_class.values())
    lines: List[str] = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"自动打标报告: {event_dir}")
    lines.append(sep)
    lines.append(f"模型: {model_name} | 置信度阈值: {conf} | 设备: {device} "
                 f"| 跟踪: {'开(ByteTrack)' if track else '关'}")
    if total_images:
        lines.append(f"图片数: {total_images} | 总框数: {total_boxes} | "
                     f"耗时: {elapsed:.1f}s ({elapsed / total_images:.2f}s/帧)")
    else:
        lines.append(f"图片数: 0 | 总框数: 0 | 耗时: {elapsed:.1f}s")

    # 每类框数分布(9 类全列,便于发现零样本)
    lines.append("")
    lines.append("[每类框数分布]")
    for cid in sorted(registry.id_to_name):
        count = per_class.get(cid, 0)
        pct = count / total_boxes * 100 if total_boxes else 0.0
        lines.append(f"  {cid} {registry.id_to_name_str(cid):<14} "
                     f"{count:>6} 个 ({pct:5.1f}%)")
    human_only = [registry.id_to_name_str(i) for i in HUMAN_ONLY_IDS
                  if i in registry.id_to_name]
    if human_only:
        lines.append(f"  ⚠️ {' / '.join(human_only)}: COCO 预训练模型"
                     f"不覆盖这 {len(human_only)} 类,需人工补标!")

    # 被跳过的 COCO 类
    if skipped:
        lines.append("")
        lines.append("[已跳过的 COCO 类](不属于项目 9 类,自动丢弃)")
        for coco_id, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            name = (coco_names or {}).get(coco_id, f"coco#{coco_id}")
            lines.append(f"  {name}: {count} 个")

    # 低置信度帧(人工优先复核)
    low = lowest_conf_frames(stats)
    if low:
        lines.append("")
        lines.append(f"[建议优先复核] 每帧平均置信度最低的 {len(low)} 帧:")
        for name, count, mc in low:
            lines.append(f"  {name}  平均置信度 {mc:.3f} ({count} 框)")

    # 零检出帧
    zeros = zero_detection_frames(stats)
    if zeros:
        lines.append("")
        lines.append(f"[零检出帧] 共 {len(zeros)} 帧没有任何检出,"
                     f"可能漏标,建议人工检查:")
        for name in zeros[:20]:
            lines.append(f"  {name}")
        if len(zeros) > 20:
            lines.append(f"  ... 其余 {len(zeros) - 20} 帧从略")

    lines.append("")
    lines.append(sep)
    lines.append("提示: 人工校正后跑 tds validate <event_dir> 做体检")
    lines.append(sep)
    return "\n".join(lines)


# ==================== ultralytics 边界(惰性 import) ====================

def _print_missing_ultralytics() -> None:
    print("❌ autolabel 需要 ultralytics(工具包本体不强制安装它)。")
    print("   请在用于打标的 Python 环境里执行: pip install ultralytics")
    print("   (torch 会作为依赖一起装上;CPU 环境即可运行)")


def _make_infer_fn(model_name: str, device: str, track: bool) -> Optional[Tuple[InferFn, Dict[int, str]]]:
    """加载模型,返回 (推理函数, COCO 类名表);ultralytics 缺失时返回 None。"""
    try:
        from ultralytics import YOLO  # 惰性 import,缺失不影响其他子命令
    except ImportError:
        return None

    model = YOLO(model_name)  # 权重不存在时 ultralytics 自动下载
    coco_names: Dict[int, str] = {int(k): str(v) for k, v in model.names.items()}

    def infer(image: Path, conf: float) -> List[Box]:
        if track:
            # 按文件名顺序逐帧跟踪(persist=True 跨帧保持轨迹,
            # 固定监控机位下减少漏标抖动);track id 丢弃,只取框和类别
            results = model.track(str(image), conf=conf, device=device,
                                  persist=True, tracker="bytetrack.yaml",
                                  verbose=False)
        else:
            results = model.predict(str(image), conf=conf, device=device,
                                    verbose=False)
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        cls_ids = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()
        xywhn = boxes.xywhn.tolist()
        return [(c, f, *xywh) for c, f, xywh in zip(cls_ids, confs, xywhn)]

    return infer, coco_names


# ==================== 编排 ====================

def autolabel(event_dir: str | Path, registry: ClassRegistry,
              conf: float = 0.2, model: str = "yolo11s.pt",
              device: str = "cpu", track: bool = False,
              infer_fn: Optional[Callable[[Path, float], List[Box]]] = None,
              coco_names: Optional[Dict[int, str]] = None) -> int:
    """自动打标主流程。返回退出码(0 成功,1 失败)。

    :param infer_fn: 测试注入口;为 None 时才真正加载 ultralytics。
    """
    event_dir = Path(event_dir)
    images_dir = event_dir / "images"
    labels_dir = event_dir / "labels"

    if not (0.0 < conf < 1.0):
        print(f"❌ --conf 必须在 (0,1) 区间,实际: {conf}")
        return 1
    if not images_dir.is_dir():
        print(f"❌ 找不到 images 文件夹: {images_dir}")
        return 1
    images = sorted(p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"❌ images 里没有图片({'/'.join(IMAGE_EXTS)}): {images_dir}")
        return 1

    if infer_fn is None:
        made = _make_infer_fn(model, device, track)
        if made is None:
            _print_missing_ultralytics()
            return 1
        infer_fn, coco_names = made

    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"找到 {len(images)} 张图片,开始自动打标"
          f"(模型 {model},conf={conf},device={device},"
          f"{'track' if track else 'predict'})...\n")

    stats: List[FrameStat] = []
    per_class: Dict[int, int] = {cid: 0 for cid in registry.id_to_name}
    skipped_total: Counter = Counter()
    t0 = time.time()

    for i, img in enumerate(images, start=1):
        lines, confs, skipped = boxes_to_lines(infer_fn(img, conf))
        skipped_total.update(skipped)
        (labels_dir / f"{img.stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        for line in lines:
            per_class[int(line.split()[0])] += 1
        stats.append((img.name, len(lines), mean_conf(confs)))
        if i % 20 == 0 or i == len(images):
            print(f"  进度 {i}/{len(images)}...")

    # classes.txt(9 类规范名,与 convert.py 行为一致)
    (labels_dir / "classes.txt").write_text(
        "\n".join(registry.names) + "\n", encoding="utf-8")

    elapsed = time.time() - t0
    report = build_report(event_dir, stats, per_class, skipped_total,
                          registry, model, conf, device, track, elapsed,
                          coco_names)
    print()
    print(report)
    report_path = labels_dir / "autolabel_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\n报告已写出: {report_path}")
    return 0

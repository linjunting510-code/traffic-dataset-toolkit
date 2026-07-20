# -*- coding: utf-8 -*-
"""tds 命令行主入口。

子命令:
- ``tds extract <输入> -o <目录>`` ffmpeg 视频抽帧(_frame_000001.jpg)
- ``tds rename <folder>``      图片批量重命名为 000001.jpg 连续编号
- ``tds autolabel <event_dir>`` 检测模型自动打标(ultralytics YOLO)
- ``tds convert <event_dir>``  X-AnyLabeling JSON -> YOLO txt
- ``tds validate <event_dir>`` 校验事件目录(配对/classes.txt/格式/统计)

不安装包时也可以这样运行(仓库根目录下)::

    set PYTHONPATH=src          :: Windows cmd
    $env:PYTHONPATH = "src"     # PowerShell
    export PYTHONPATH=src       # bash
    python -m traffic_dataset.cli --help
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .classes import load_classes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tds",
        description="交通事件视觉数据集构建工具包"
        "(抽帧图片重命名 / X-AnyLabeling JSON 转 YOLO / 数据集校验)",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="<子命令>")

    # ---- rename ----
    p_rename = sub.add_parser(
        "rename",
        help="图片批量重命名为 000001.jpg 起的连续编号",
        description="把文件夹里的图片(jpg/jpeg/png)按文件名排序后,"
        "重命名为 000001.jpg 起的连续编号。两步改名防冲突。",
    )
    p_rename.add_argument("folder", help="图片所在文件夹")
    p_rename.add_argument("--dry-run", action="store_true",
                          help="只预览改名计划,不实际改动文件")
    p_rename.add_argument("--start", type=int, default=1,
                          help="起始编号(默认 1)")
    p_rename.add_argument("--digits", type=int, default=6,
                          help="编号位数(默认 6,即 000001)")

    # ---- convert ----
    p_convert = sub.add_parser(
        "convert",
        help="X-AnyLabeling JSON 标注转 YOLO txt",
        description="读取 <事件目录>/images/*.json,按 configs/classes.yaml 的"
        "9 类定义转换为 YOLO 格式,写入 <事件目录>/labels/*.txt;"
        "不在 9 类里的目标自动跳过并统计。",
    )
    p_convert.add_argument("event_dir",
                           help="事件目录(内含 images/ 子目录)")
    p_convert.add_argument("--classes", default=None, metavar="YAML",
                           help="自定义 classes.yaml 路径"
                           "(默认用仓库 configs/classes.yaml,"
                           "也可用环境变量 TDS_CLASSES_YAML)")

    # ---- validate ----
    p_validate = sub.add_parser(
        "validate",
        help="校验事件目录的图片/标注配对、classes.txt 与标注格式",
        description="校验 <事件目录>(images/ + labels/):图片-标注配对、"
        "classes.txt 一致性、标注行格式、每类框数统计。"
        "有错误返回退出码 1,仅警告返回 0。",
    )
    p_validate.add_argument("event_dir",
                            help="事件目录(内含 images/ 与 labels/ 子目录)")
    p_validate.add_argument("--classes", default=None, metavar="YAML",
                            help="自定义 classes.yaml 路径(同上)")

    # ---- extract ----
    from .extract import (DEFAULT_TARGET_FRAMES, OUTPUT_EXTS,
                          VIDEO_EXTS)  # 仅取常量,避免顶层硬依赖
    p_extract = sub.add_parser(
        "extract",
        help="ffmpeg 视频抽帧为 <视频名>_frame_000001.jpg 连续编号图片",
        description="用 ffmpeg 把视频抽成连续编号图片帧"
        "(命名:<视频文件名去后缀>_frame_000001.jpg,6 位编号)。"
        "输入可以是单个视频或目录(不递归,自动跳过 .part 未完成下载文件)。"
        "需要本机已安装 ffmpeg / ffprobe。",
    )
    p_extract.add_argument("input",
                           help="视频文件,或包含视频的目录(不递归;"
                           f"支持 {'/'.join(VIDEO_EXTS)})")
    p_extract.add_argument("-o", "--output", required=True,
                           help="帧图片输出目录(不存在会自动创建)")
    mode = p_extract.add_mutually_exclusive_group()
    mode.add_argument("--fps", type=float, default=None, metavar="N",
                      help="固定帧率抽帧(帧/秒)")
    mode.add_argument("--frames", type=int, default=None, metavar="M",
                      help="目标总帧数模式:先 ffprobe 测时长,"
                      "fps = M / 时长,实现'短视频密采、长视频稀采'。"
                      f"与 --fps 都不给时默认 M={DEFAULT_TARGET_FRAMES}"
                      "(约能覆盖一个事件片段全过程,标注量可控)")
    p_extract.add_argument("--ext", default="jpg", choices=list(OUTPUT_EXTS),
                           help="输出图片格式(默认 jpg)")

    # ---- autolabel ----
    p_auto = sub.add_parser(
        "autolabel",
        help="检测模型自动打标(ultralytics YOLO,COCO 6 类映射到项目 9 类)",
        description="对 <事件目录>/images/* 跑 ultralytics YOLO(COCO 预训练),"
        "把 person/car/motorcycle/bus/truck/traffic light 六类检出框"
        "映射为项目 9 类写入 labels/*.txt,并生成抽检报告"
        "(低置信度帧/零检出帧/类别分布)。"
        "Traffic Cone/Barrier/Tree 三类 COCO 不覆盖,需人工补标。"
        "需要在打标环境安装 ultralytics(工具包本体不硬依赖)。",
    )
    p_auto.add_argument("event_dir", help="事件目录(内含 images/ 子目录)")
    p_auto.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值(默认 0.25)")
    p_auto.add_argument("--model", default="yolo11n.pt", metavar="权重",
                        help="ultralytics 模型权重(默认 yolo11n.pt,"
                        "不存在时自动下载)")
    p_auto.add_argument("--device", default="cpu",
                        help="推理设备(默认 cpu;有显卡可填 0 等)")
    p_auto.add_argument("--track", action="store_true",
                        help="启用 ByteTrack 逐帧跟踪(persist=True,"
                        "固定监控机位下减少漏标抖动;track id 丢弃)")
    p_auto.add_argument("--classes", default=None, metavar="YAML",
                        help="自定义 classes.yaml 路径(同上)")
    return parser


def main(argv=None) -> int:
    """CLI 入口,返回退出码(console_scripts 会包装成 sys.exit)。"""
    args = build_parser().parse_args(argv)

    if args.command == "rename":
        from .rename import rename_folder
        return rename_folder(args.folder, dry_run=args.dry_run,
                             start=args.start, digits=args.digits)
    if args.command == "extract":
        from .extract import extract
        return extract(args.input, args.output, fps=args.fps,
                       frames=args.frames, ext=args.ext)

    # convert / validate 都需要类别定义
    try:
        registry = load_classes(getattr(args, "classes", None))
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ 加载类别定义失败: {exc}")
        return 1

    if args.command == "convert":
        from .convert import convert_event
        return convert_event(args.event_dir, registry)
    if args.command == "autolabel":
        from .autolabel import autolabel
        return autolabel(args.event_dir, registry, conf=args.conf,
                         model=args.model, device=args.device,
                         track=args.track)
    if args.command == "validate":
        from .validate import validate_event
        return validate_event(args.event_dir, registry)

    return 2  # 理论上到不了(argparse required=True)


if __name__ == "__main__":
    sys.exit(main())

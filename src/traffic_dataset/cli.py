# -*- coding: utf-8 -*-
"""tds 命令行主入口。

子命令:
- ``tds rename <folder>``      图片批量重命名为 000001.jpg 连续编号
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
    return parser


def main(argv=None) -> int:
    """CLI 入口,返回退出码(console_scripts 会包装成 sys.exit)。"""
    args = build_parser().parse_args(argv)

    if args.command == "rename":
        from .rename import rename_folder
        return rename_folder(args.folder, dry_run=args.dry_run,
                             start=args.start, digits=args.digits)

    # convert / validate 都需要类别定义
    try:
        registry = load_classes(getattr(args, "classes", None))
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ 加载类别定义失败: {exc}")
        return 1

    if args.command == "convert":
        from .convert import convert_event
        return convert_event(args.event_dir, registry)
    if args.command == "validate":
        from .validate import validate_event
        return validate_event(args.event_dir, registry)

    return 2  # 理论上到不了(argparse required=True)


if __name__ == "__main__":
    sys.exit(main())

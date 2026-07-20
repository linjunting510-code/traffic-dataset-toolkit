# -*- coding: utf-8 -*-
"""ffmpeg 视频抽帧:把事故/路况视频抽成连续编号的图片帧。

命名沿用项目既有惯例:``<视频文件名去后缀>_frame_000001.jpg``(6 位编号)。

两种采样模式(互斥):

- ``--fps N``   :固定帧率抽帧(帧/秒)
- ``--frames M``:目标总帧数模式 —— 先用 ffprobe 拿视频时长,
  ``fps = M / 时长``。这就是"采样帧率按事件时长决定
  (短视频密采、长视频稀采)"的自动化。两个参数都不给时默认按 120 帧目标。

设计约定:纯逻辑(时长解析、fps 计算、命名生成、视频文件筛选)
与 subprocess 调用(ffmpeg/ffprobe)分离,便于单元测试。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

# 支持的视频格式(小写后缀)
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".ts", ".flv", ".wmv", ".webm")
# --frames 默认目标帧数:一个交通事件片段抽 120 帧左右,
# 既能覆盖事件全过程,标注量又不至于爆炸
DEFAULT_TARGET_FRAMES = 120
# 帧编号位数(沿用项目既有命名:..._frame_000001.jpg)
FRAME_DIGITS = 6
# 未完成下载的临时文件后缀(批处理时跳过并提醒)
PART_SUFFIX = ".part"
# 允许输出的图片格式
OUTPUT_EXTS = ("jpg", "jpeg", "png")


# ==================== 纯逻辑(不碰 subprocess,便于单测) ====================

def is_video_file(name: str) -> bool:
    """判断文件名是不是支持的视频格式(大小写不敏感)。"""
    return Path(name).suffix.lower() in VIDEO_EXTS


def find_videos(folder: Path) -> Tuple[List[Path], List[Path]]:
    """扫描目录(不递归),返回 (视频文件列表, .part 未完成文件列表)。

    两个列表都按文件名排序,保证处理顺序稳定。
    """
    videos: List[Path] = []
    parts: List[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.name.lower().endswith(PART_SUFFIX):
            parts.append(p)  # 未完成下载(如 xxx.mp4.part),跳过
        elif is_video_file(p.name):
            videos.append(p)
    return videos, parts


def frame_stem(video_name: str) -> str:
    """帧文件名主干:'clip001.mp4' -> 'clip001_frame'。"""
    return f"{Path(video_name).stem}_frame"


def frame_name(video_name: str, index: int, ext: str = "jpg") -> str:
    """单个帧文件名:'clip001.mp4' 第 1 帧 -> 'clip001_frame_000001.jpg'。"""
    return f"{frame_stem(video_name)}_{index:0{FRAME_DIGITS}d}.{ext}"


def build_output_pattern(out_dir: Path, video_name: str, ext: str = "jpg") -> str:
    r"""ffmpeg 输出模式串,如 'out\clip001_frame_%06d.jpg'。"""
    return str(out_dir / f"{frame_stem(video_name)}_%0{FRAME_DIGITS}d.{ext}")


def compute_fps(duration_sec: float, target_frames: int) -> float:
    """目标总帧数模式:fps = 目标帧数 / 视频时长(秒)。"""
    if duration_sec <= 0:
        raise ValueError(f"视频时长必须为正数,实际: {duration_sec}")
    if target_frames <= 0:
        raise ValueError(f"目标帧数必须为正整数,实际: {target_frames}")
    return target_frames / duration_sec


def estimate_frames(duration_sec: float, fps: float) -> int:
    """固定帧率模式下估算产出帧数(实际数量以抽帧结果为准)。"""
    return max(1, round(duration_sec * fps))


def parse_duration(text: str) -> float:
    """解析 ffprobe 输出的时长(秒)。无法解析时抛 ValueError。"""
    try:
        value = float(text.strip())
    except (TypeError, ValueError):
        raise ValueError(f"无法解析视频时长: {text!r}")
    if value <= 0:
        raise ValueError(f"视频时长必须为正数,实际: {value}")
    return value


def already_extracted(out_dir: Path, video_name: str, ext: str = "jpg") -> bool:
    """输出目录里已有该视频的抽帧结果(防重名覆盖,用于跳过)。"""
    return any(out_dir.glob(f"{frame_stem(video_name)}_*.{ext}"))


def count_extracted(out_dir: Path, video_name: str, ext: str = "jpg") -> int:
    """统计输出目录里某个视频实际抽出的帧数。"""
    return sum(1 for _ in out_dir.glob(f"{frame_stem(video_name)}_*.{ext}"))


# ==================== subprocess 边界(ffmpeg / ffprobe) ====================

def find_tool(name: str) -> Optional[str]:
    """在 PATH 上查找可执行程序,找不到返回 None。"""
    return shutil.which(name)


def probe_duration(video: Path, ffprobe: str = "ffprobe") -> float:
    """用 ffprobe 获取视频时长(秒)。失败抛 RuntimeError。"""
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe 执行失败")
    try:
        return parse_duration(result.stdout)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def run_ffmpeg(video: Path, out_pattern: str, fps: float,
               ffmpeg: str = "ffmpeg") -> None:
    """调 ffmpeg 按指定帧率抽帧。失败抛 RuntimeError。"""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps={fps:.6f}",
    ]
    # jpg 输出给固定高质量(-q:v 2),避免默认画质太糊
    if out_pattern.lower().endswith((".jpg", ".jpeg")):
        cmd += ["-q:v", "2"]
    cmd.append(out_pattern)
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg 执行失败")


# ==================== 编排 ====================

def _print_missing_tools_help() -> None:
    print("❌ 未找到 ffmpeg / ffprobe。抽帧功能依赖它们,请先安装:")
    print("   - winget install ffmpeg")
    print("   - 或从官网下载 Windows 构建: https://ffmpeg.org/download.html")
    print("   安装后重开终端,确认 ffmpeg -version 可用再重试。")


def extract(input_path: str | Path, output_dir: str | Path,
            fps: Optional[float] = None, frames: Optional[int] = None,
            ext: str = "jpg") -> int:
    """抽帧主流程。返回退出码(0 成功,1 有失败/无可处理视频)。"""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    ext = ext.lower().lstrip(".")

    # ---- 参数校验 ----
    if ext not in OUTPUT_EXTS:
        print(f"❌ 不支持的输出格式: {ext}(可选 {'/'.join(OUTPUT_EXTS)})")
        return 1
    if fps is not None and fps <= 0:
        print(f"❌ --fps 必须为正数,实际: {fps}")
        return 1
    if frames is not None and frames <= 0:
        print(f"❌ --frames 必须为正整数,实际: {frames}")
        return 1

    # ---- 收集输入视频 ----
    if not input_path.exists():
        print(f"❌ 输入不存在: {input_path}")
        return 1
    if input_path.is_file():
        if input_path.name.lower().endswith(PART_SUFFIX):
            print(f"❌ 输入是未完成下载的 .part 文件: {input_path.name}")
            print("   请等下载完成后再抽帧。")
            return 1
        if not is_video_file(input_path.name):
            print(f"❌ 不是支持的视频格式({'/'.join(VIDEO_EXTS)}): {input_path.name}")
            return 1
        videos, part_files = [input_path], []
    else:
        videos, part_files = find_videos(input_path)
        for pf in part_files:
            print(f"⚠️ 跳过未完成下载的文件: {pf.name}")
        if not videos:
            print(f"❌ 目录里没有可处理的视频({ '/'.join(VIDEO_EXTS) },不递归):"
                  f" {input_path}")
            return 1

    # ---- 检查 ffmpeg / ffprobe ----
    if not find_tool("ffmpeg") or not find_tool("ffprobe"):
        _print_missing_tools_help()
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 采样模式 ----
    # fps 给了用固定帧率;否则用目标总帧数模式(默认 120)
    target = frames if frames is not None else DEFAULT_TARGET_FRAMES
    mode_desc = (f"--fps {fps}(固定帧率)" if fps is not None
                 else f"--frames {target}(按视频时长自动算帧率)")

    # ---- 打印抽帧计划 ----
    print(f"\n抽帧计划({mode_desc})")
    print(f"输出目录: {output_dir}\n")
    plans: List[Tuple[Path, float, float, int]] = []  # (视频, 时长, 采用fps, 预计帧数)
    failures = 0
    for i, video in enumerate(videos, start=1):
        try:
            duration = probe_duration(video)
        except RuntimeError as exc:
            print(f"  ❌ {video.name}: 读取时长失败({exc}),已跳过")
            failures += 1
            continue
        use_fps = fps if fps is not None else compute_fps(duration, target)
        est = estimate_frames(duration, use_fps)
        plans.append((video, duration, use_fps, est))
        print(f"  {i}. {video.name}")
        print(f"     时长 {duration:.1f}s | 采用 fps={use_fps:.3f} | 预计 {est} 帧")

    if not plans:
        print("\n❌ 没有可处理的视频。")
        return 1

    # ---- 逐个抽帧 ----
    print("\n开始抽帧...")
    done_videos = 0
    total_frames = 0
    for video, duration, use_fps, est in plans:
        if already_extracted(output_dir, video.name, ext):
            print(f"  ⚠️ 跳过 {video.name}: 输出目录已存在 "
                  f"{frame_stem(video.name)}_*.{ext},为避免覆盖不重抽")
            continue
        pattern = build_output_pattern(output_dir, video.name, ext)
        print(f"  ▶ {video.name} ...", end=" ", flush=True)
        try:
            run_ffmpeg(video, pattern, use_fps)
        except RuntimeError as exc:
            print(f"失败({exc})")
            failures += 1
            continue
        count = count_extracted(output_dir, video.name, ext)
        total_frames += count
        done_videos += 1
        print(f"完成,抽出 {count} 帧")

    # ---- 统计 ----
    print(f"\n{'=' * 50}")
    print(f"抽帧完成: 成功 {done_videos}/{len(plans)} 个视频,"
          f"共 {total_frames} 帧,输出在 {output_dir}")
    if failures:
        print(f"⚠️ 有 {failures} 个视频处理失败,见上方错误信息。")
    print(f"{'=' * 50}")
    return 1 if failures else 0

# -*- coding: utf-8 -*-
"""extract.py 的单元测试(标准库 unittest + mock,不依赖真实 ffmpeg)。

运行方式(仓库根目录下)::

    set PYTHONPATH=src
    python -m unittest discover tests
"""
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from traffic_dataset import extract as ex


class TestNaming(unittest.TestCase):
    """帧文件命名:沿用 <视频名去后缀>_frame_000001.jpg 惯例。"""

    def test_frame_stem(self):
        self.assertEqual(ex.frame_stem("clip001.mp4"), "clip001_frame")
        self.assertEqual(ex.frame_stem("G0015_hd_m3u8_20250930.mkv"),
                         "G0015_hd_m3u8_20250930_frame")

    def test_frame_name_six_digits(self):
        self.assertEqual(ex.frame_name("clip001.mp4", 1),
                         "clip001_frame_000001.jpg")
        self.assertEqual(ex.frame_name("clip001.mp4", 123),
                         "clip001_frame_000123.jpg")

    def test_frame_name_other_ext(self):
        self.assertEqual(ex.frame_name("a b.mkv", 7, "png"),
                         "a b_frame_000007.png")

    def test_output_pattern(self):
        pattern = ex.build_output_pattern(Path("out"), "clip001.mp4")
        self.assertIn("clip001_frame_%06d.jpg", pattern)


class TestFpsCalc(unittest.TestCase):
    """目标总帧数模式:fps = 目标帧数 / 时长。"""

    def test_frames_over_duration(self):
        self.assertAlmostEqual(ex.compute_fps(10.0, 120), 12.0)
        self.assertAlmostEqual(ex.compute_fps(61.2, 120), 120 / 61.2)

    def test_short_video_dense_sampling(self):
        """短视频密采:5 秒视频抽 120 帧 -> 24 fps。"""
        self.assertAlmostEqual(ex.compute_fps(5.0, 120), 24.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ex.compute_fps(0, 120)      # 时长为 0
        with self.assertRaises(ValueError):
            ex.compute_fps(-3, 120)     # 负时长
        with self.assertRaises(ValueError):
            ex.compute_fps(10, 0)       # 目标帧数为 0

    def test_estimate_frames(self):
        self.assertEqual(ex.estimate_frames(10.0, 2.5), 25)
        self.assertEqual(ex.estimate_frames(0.1, 0.5), 1)  # 至少 1 帧


class TestParseDuration(unittest.TestCase):
    """ffprobe 时长输出解析。"""

    def test_normal(self):
        self.assertAlmostEqual(ex.parse_duration("12.345\n"), 12.345)
        self.assertAlmostEqual(ex.parse_duration("  61.2  "), 61.2)

    def test_invalid(self):
        for bad in ("", "N/A", "abc"):
            with self.assertRaises(ValueError):
                ex.parse_duration(bad)


class TestFindVideos(unittest.TestCase):
    """目录扫描:视频筛选、.part 过滤、不递归。"""

    def _touch(self, folder: Path, *names: str) -> None:
        for name in names:
            (folder / name).write_bytes(b"")

    def test_filter_and_part_skip(self):
        with TemporaryDirectory() as d:
            folder = Path(d)
            self._touch(folder, "b.avi", "a.mp4", "c.txt", "x.jpg",
                        "dl1.mp4.part", "dl2.part")
            videos, parts = ex.find_videos(folder)
            self.assertEqual([v.name for v in videos], ["a.mp4", "b.avi"])
            self.assertEqual([p.name for p in parts], ["dl1.mp4.part", "dl2.part"])

    def test_case_insensitive_ext(self):
        self.assertTrue(ex.is_video_file("A.MP4"))
        self.assertTrue(ex.is_video_file("b.MkV"))
        self.assertFalse(ex.is_video_file("c.mp4.part"))  # .part 不是视频
        self.assertFalse(ex.is_video_file("d.txt"))

    def test_not_recursive(self):
        with TemporaryDirectory() as d:
            folder = Path(d)
            sub = folder / "sub"
            sub.mkdir()
            (sub / "inner.mp4").write_bytes(b"")
            videos, parts = ex.find_videos(folder)
            self.assertEqual(videos, [])
            self.assertEqual(parts, [])

    def test_empty_dir(self):
        with TemporaryDirectory() as d:
            videos, parts = ex.find_videos(Path(d))
            self.assertEqual(videos, [])
            self.assertEqual(parts, [])


class TestAlreadyExtracted(unittest.TestCase):
    """防重名覆盖:输出目录已有同名帧则跳过。"""

    def test_exists_and_absent(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            self.assertFalse(ex.already_extracted(out, "clip001.mp4"))
            (out / "clip001_frame_000001.jpg").write_bytes(b"")
            self.assertTrue(ex.already_extracted(out, "clip001.mp4"))
            # 别的视频不受影响
            self.assertFalse(ex.already_extracted(out, "clip002.mp4"))
            # 扩展名不同也不算冲突
            self.assertFalse(ex.already_extracted(out, "clip001.mp4", "png"))

    def test_count_extracted(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            for i in range(1, 4):
                (out / ex.frame_name("clip001.mp4", i)).write_bytes(b"")
            self.assertEqual(ex.count_extracted(out, "clip001.mp4"), 3)


class TestProbeDurationMock(unittest.TestCase):
    """ffprobe 边界:mock subprocess.run,不需要真实 ffprobe。"""

    def test_success(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="12.340\n", stderr="")
        with mock.patch.object(ex.subprocess, "run", return_value=completed):
            self.assertAlmostEqual(ex.probe_duration(Path("x.mp4")), 12.34)

    def test_failure_raises(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="moov atom not found")
        with mock.patch.object(ex.subprocess, "run", return_value=completed):
            with self.assertRaises(RuntimeError) as ctx:
                ex.probe_duration(Path("broken.mp4"))
            self.assertIn("moov atom not found", str(ctx.exception))

    def test_unparseable_output_raises(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="N/A\n", stderr="")
        with mock.patch.object(ex.subprocess, "run", return_value=completed):
            with self.assertRaises(RuntimeError):
                ex.probe_duration(Path("x.mp4"))


class TestExtractGuardrails(unittest.TestCase):
    """编排层的护栏:ffmpeg 缺失 / 非法参数 / 空目录(不碰真实 ffmpeg)。"""

    def test_missing_ffmpeg_returns_1(self):
        with TemporaryDirectory() as d:
            video = Path(d) / "clip001.mp4"
            video.write_bytes(b"")  # 文件存在,走不到真实 ffmpeg
            with mock.patch.object(ex.shutil, "which", return_value=None):
                self.assertEqual(ex.extract(video, Path(d) / "out"), 1)

    def test_part_file_input_returns_1(self):
        with TemporaryDirectory() as d:
            part = Path(d) / "movie.mp4.part"
            part.write_bytes(b"")
            self.assertEqual(ex.extract(part, Path(d) / "out"), 1)

    def test_dir_without_videos_returns_1(self):
        with TemporaryDirectory() as d:
            (Path(d) / "readme.txt").write_bytes(b"")
            self.assertEqual(ex.extract(d, Path(d) / "out"), 1)

    def test_bad_params_return_1(self):
        with TemporaryDirectory() as d:
            video = Path(d) / "clip001.mp4"
            video.write_bytes(b"")
            self.assertEqual(ex.extract(video, Path(d) / "o", fps=0), 1)
            self.assertEqual(ex.extract(video, Path(d) / "o", frames=-5), 1)
            self.assertEqual(ex.extract(video, Path(d) / "o", ext="gif"), 1)

    def test_nonexistent_input_returns_1(self):
        with TemporaryDirectory() as d:
            self.assertEqual(
                ex.extract(Path(d) / "nope.mp4", Path(d) / "out"), 1)


class TestExtractFlowMock(unittest.TestCase):
    """全流程编排:mock 掉 probe/run,验证计划、跳过与统计逻辑。"""

    def _fake_run(self, out_dir: Path):
        """伪造 subprocess.run:ffprobe 返回固定时长;ffmpeg 写出假帧文件。"""
        def _run(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout="10.0\n", stderr="")
            # ffmpeg:最后一个参数是输出模式 ..._frame_%06d.jpg
            pattern = cmd[-1]
            for i in range(1, 6):  # 假装抽出 5 帧
                Path(pattern.replace("%06d", f"{i:06d}")).write_bytes(b"")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr="")
        return _run

    def test_frames_mode_flow(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "clip001.mp4").write_bytes(b"")
            out = root / "out"
            with mock.patch.object(ex.shutil, "which", side_effect=lambda n: n), \
                 mock.patch.object(ex.subprocess, "run",
                                   side_effect=self._fake_run(out)):
                # 时长 10s、目标 120 帧 -> fps 应为 12
                self.assertEqual(ex.extract(root, out, frames=120), 0)
            self.assertEqual(ex.count_extracted(out, "clip001.mp4"), 5)

    def test_skip_already_extracted(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "clip001.mp4").write_bytes(b"")
            out = root / "out"
            out.mkdir()
            # 预置一帧,触发"已有结果跳过"
            (out / ex.frame_name("clip001.mp4", 1)).write_bytes(b"")
            with mock.patch.object(ex.shutil, "which", side_effect=lambda n: n), \
                 mock.patch.object(ex.subprocess, "run",
                                   side_effect=self._fake_run(out)) as mrun:
                self.assertEqual(ex.extract(root, out), 0)
            # ffmpeg 不应被调用(只有 ffprobe 一次)
            self.assertEqual(
                [c.args[0][0] for c in mrun.call_args_list], ["ffprobe"])


if __name__ == "__main__":
    unittest.main()

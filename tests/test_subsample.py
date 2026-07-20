# -*- coding: utf-8 -*-
"""subsample.py 的单元测试(标准库 unittest,纯文件操作不需要 mock)。

运行方式(仓库根目录下)::

    set PYTHONPATH=src
    python -m unittest discover tests
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from traffic_dataset import subsample as ss


class TestSelectIndices(unittest.TestCase):
    """选帧索引计算(纯逻辑)。"""

    def test_every_5_from_0(self):
        idx = ss.select_indices(120, 5, 0)
        self.assertEqual(idx, list(range(0, 120, 5)))  # 0,5,...,115
        self.assertEqual(len(idx), 24)

    def test_every_1_keeps_all(self):
        self.assertEqual(ss.select_indices(10, 1, 0), list(range(10)))

    def test_offset(self):
        self.assertEqual(ss.select_indices(12, 5, 2), [2, 7])
        self.assertEqual(ss.select_indices(10, 3, 1), [1, 4, 7])

    def test_offset_beyond_total(self):
        self.assertEqual(ss.select_indices(5, 5, 7), [])

    def test_empty_source(self):
        self.assertEqual(ss.select_indices(0, 5, 0), [])

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            ss.select_indices(10, 0)      # every < 1
        with self.assertRaises(ValueError):
            ss.select_indices(10, 5, -1)  # offset < 0


class _EventFixture:
    """在临时目录造一个事件目录:10 张图 + 部分标注。"""

    def __init__(self, root: Path, with_labels: bool = True):
        self.event = root / "src_event"
        images = self.event / "images"
        images.mkdir(parents=True)
        for i in range(1, 11):  # 000001.jpg ... 000010.jpg
            (images / f"{i:06d}.jpg").write_bytes(b"img")
        if with_labels:
            labels = self.event / "labels"
            labels.mkdir()
            # 只给 1,3,6 号帧写标注(制造"有图无标注"场景)
            for i in (1, 3, 6):
                (labels / f"{i:06d}.txt").write_text("0 0.5 0.5 0.1 0.1\n",
                                                     encoding="utf-8")
            (labels / "classes.txt").write_text("Car\n", encoding="utf-8")
            (labels / "autolabel_report.txt").write_text("旧报告\n",
                                                         encoding="utf-8")


class TestSubsampleFlow(unittest.TestCase):
    """文件复制编排:配对、提示、防覆盖。"""

    def test_images_copied_with_original_names(self):
        with TemporaryDirectory() as d:
            fx = _EventFixture(Path(d), with_labels=False)
            out = Path(d) / "sub"
            self.assertEqual(ss.subsample(fx.event, out, every=5), 0)
            copied = sorted(p.name for p in (out / "images").iterdir())
            # every=5,offset=0 -> 索引 0,5 -> 000001.jpg,000006.jpg
            self.assertEqual(copied, ["000001.jpg", "000006.jpg"])

    def test_labels_paired_and_meta_filtered(self):
        with TemporaryDirectory() as d:
            fx = _EventFixture(Path(d), with_labels=True)
            out = Path(d) / "sub"
            self.assertEqual(ss.subsample(fx.event, out, every=5), 0)
            labels = out / "labels"
            # 选中的 000001/000006 都有标注;classes.txt 复制;
            # autolabel_report.txt 不复制
            self.assertEqual(sorted(p.name for p in labels.iterdir()),
                             ["000001.txt", "000006.txt", "classes.txt"])
            self.assertEqual((labels / "000001.txt").read_text(encoding="utf-8"),
                             "0 0.5 0.5 0.1 0.1\n")

    def test_missing_label_warns_but_copies_image(self):
        with TemporaryDirectory() as d:
            fx = _EventFixture(Path(d), with_labels=True)
            out = Path(d) / "sub"
            # every=3,offset=1 -> 索引 1,4,7,10 -> 000002/000005/000008(无标注)
            # 与 000011 不存在 -> 实际 000002,000005,000008,000010? 共 10 张
            # 索引 1,4,7 -> 000002,000005,000008;索引 10 越界 -> 3 张
            self.assertEqual(ss.subsample(fx.event, out, every=3, offset=1), 0)
            copied = sorted(p.name for p in (out / "images").iterdir())
            self.assertEqual(copied, ["000002.jpg", "000005.jpg", "000008.jpg"])
            # 这三张都没有标注 -> labels 目录只有 classes.txt
            labels = out / "labels"
            self.assertEqual(sorted(p.name for p in labels.iterdir()),
                             ["classes.txt"])

    def test_nonempty_output_refused(self):
        with TemporaryDirectory() as d:
            fx = _EventFixture(Path(d))
            out = Path(d) / "sub"
            out.mkdir()
            (out / "existing.txt").write_text("别动我", encoding="utf-8")
            self.assertEqual(ss.subsample(fx.event, out, every=5), 1)
            # 原有文件未被碰
            self.assertTrue((out / "existing.txt").is_file())
            self.assertFalse((out / "images").exists())

    def test_force_overwrites(self):
        with TemporaryDirectory() as d:
            fx = _EventFixture(Path(d), with_labels=False)
            out = Path(d) / "sub"
            out.mkdir()
            (out / "existing.txt").write_text("旧文件", encoding="utf-8")
            self.assertEqual(ss.subsample(fx.event, out, every=5, force=True), 0)
            self.assertTrue((out / "images" / "000001.jpg").is_file())
            # 非同名旧文件保留
            self.assertTrue((out / "existing.txt").is_file())

    def test_offset_shifts_selection(self):
        with TemporaryDirectory() as d:
            fx = _EventFixture(Path(d), with_labels=False)
            out = Path(d) / "sub"
            self.assertEqual(ss.subsample(fx.event, out, every=5, offset=2), 0)
            copied = sorted(p.name for p in (out / "images").iterdir())
            # 索引 2,7 -> 000003.jpg,000008.jpg
            self.assertEqual(copied, ["000003.jpg", "000008.jpg"])

    def test_bad_params_and_missing_dirs(self):
        with TemporaryDirectory() as d:
            fx = _EventFixture(Path(d), with_labels=False)
            out = Path(d) / "sub"
            self.assertEqual(ss.subsample(fx.event, out, every=0), 1)
            self.assertEqual(ss.subsample(fx.event, out, every=5, offset=-1), 1)
            self.assertEqual(
                ss.subsample(Path(d) / "no_such", out, every=5), 1)
            # offset 超过帧数 -> 选帧为空
            self.assertEqual(ss.subsample(fx.event, out, every=5, offset=99), 1)


if __name__ == "__main__":
    unittest.main()

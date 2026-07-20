# -*- coding: utf-8 -*-
"""autolabel.py 的单元测试(标准库 unittest + mock,不依赖 ultralytics)。

运行方式(仓库根目录下)::

    set PYTHONPATH=src
    python -m unittest discover tests
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from traffic_dataset import autolabel as al
from traffic_dataset.classes import load_classes


def _box(coco_cls, conf, x=0.5, y=0.5, w=0.1, h=0.1):
    """构造一个检出框:(coco_cls, conf, x, y, w, h)。"""
    return (coco_cls, conf, x, y, w, h)


class TestCocoMapping(unittest.TestCase):
    """COCO → 项目 9 类映射。"""

    def test_mapped_classes(self):
        self.assertEqual(al.map_coco_class(0), 4)   # person -> Pedestrian
        self.assertEqual(al.map_coco_class(2), 0)   # car -> Car
        self.assertEqual(al.map_coco_class(3), 3)   # motorcycle -> Motorcycle
        self.assertEqual(al.map_coco_class(5), 1)   # bus -> Bus
        self.assertEqual(al.map_coco_class(7), 2)   # truck -> Truck
        self.assertEqual(al.map_coco_class(9), 8)   # traffic light -> Traffic light

    def test_unmapped_returns_none(self):
        for coco_id in (1, 4, 6, 8, 13, 24, 57, 79):  # bicycle/airplane/...
            self.assertIsNone(al.map_coco_class(coco_id))

    def test_exactly_six_mapped(self):
        self.assertEqual(len(al.COCO_TO_PROJECT), 6)


class TestBoxesToLines(unittest.TestCase):
    """检出框 → YOLO 行:映射、跳过统计、坐标截断、置信度保留。"""

    def test_mixed_boxes(self):
        lines, confs, skipped = al.boxes_to_lines([
            _box(2, 0.9),           # car -> 保留
            _box(1, 0.8),           # bicycle -> 跳过
            _box(0, 0.7),           # person -> 保留
            _box(24, 0.6),          # backpack -> 跳过
        ])
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("0 "))   # Car
        self.assertTrue(lines[1].startswith("4 "))   # Pedestrian
        self.assertEqual(confs, [0.9, 0.7])
        self.assertEqual(skipped[1], 1)
        self.assertEqual(skipped[24], 1)

    def test_line_format_six_decimals(self):
        lines, _, _ = al.boxes_to_lines([_box(2, 0.9, 0.123456789, 0.5, 0.1, 0.2)])
        self.assertEqual(lines[0], "0 0.123457 0.500000 0.100000 0.200000")

    def test_coords_clamped_to_unit(self):
        # 模型偶尔输出轻微越界值,应截断到 [0,1] 而不是留坑给 validate
        lines, _, _ = al.boxes_to_lines([_box(2, 0.9, 1.0003, -0.0001, 0.5, 0.5)])
        self.assertEqual(lines[0], "0 1.000000 0.000000 0.500000 0.500000")

    def test_degenerate_box_dropped(self):
        lines, confs, _ = al.boxes_to_lines([_box(2, 0.9, w=0.0)])
        self.assertEqual(lines, [])
        self.assertEqual(confs, [])


class TestFrameStats(unittest.TestCase):
    """置信度统计:低置信帧排序、零检出帧识别。"""

    def setUp(self):
        # (图片名, 框数, 平均置信度)
        self.stats = [
            ("000001.jpg", 5, 0.80),
            ("000002.jpg", 0, None),   # 零检出
            ("000003.jpg", 3, 0.30),   # 最低
            ("000004.jpg", 2, 0.50),
            ("000005.jpg", 0, None),   # 零检出
            ("000006.jpg", 1, 0.40),
        ]

    def test_lowest_conf_sorted_ascending(self):
        low = al.lowest_conf_frames(self.stats, n=3)
        self.assertEqual([s[0] for s in low],
                         ["000003.jpg", "000006.jpg", "000004.jpg"])

    def test_zero_frames_excluded_from_low_conf(self):
        low = al.lowest_conf_frames(self.stats, n=10)
        self.assertNotIn("000002.jpg", [s[0] for s in low])
        self.assertEqual(len(low), 4)  # 只有 4 帧有框

    def test_zero_detection_frames(self):
        self.assertEqual(al.zero_detection_frames(self.stats),
                         ["000002.jpg", "000005.jpg"])

    def test_mean_conf(self):
        self.assertAlmostEqual(al.mean_conf([0.5, 0.7]), 0.6)
        self.assertIsNone(al.mean_conf([]))


class TestBuildReport(unittest.TestCase):
    """报告生成:关键小节齐全。"""

    @classmethod
    def setUpClass(cls):
        cls.registry = load_classes()

    def test_report_sections(self):
        stats = [("000001.jpg", 2, 0.9), ("000002.jpg", 0, None)]
        per_class = {cid: 0 for cid in self.registry.id_to_name}
        per_class[0] = 2
        skipped = {1: 3}  # bicycle x3
        report = al.build_report(
            Path("evt"), stats, per_class, skipped, self.registry,
            "yolo11n.pt", 0.25, "cpu", False, 1.5,
            coco_names={1: "bicycle"})
        self.assertIn("自动打标报告", report)
        self.assertIn("每类框数分布", report)
        self.assertIn("Car", report)
        self.assertIn("需人工补标", report)          # 三类 COCO 不覆盖提醒
        self.assertIn("bicycle: 3 个", report)       # 跳过统计带 COCO 名
        self.assertIn("零检出帧", report)
        self.assertIn("000002.jpg", report)
        self.assertIn("建议优先复核", report)

    def test_report_mentions_human_only_classes(self):
        report = al.build_report(
            Path("evt"), [], {}, {}, self.registry,
            "m.pt", 0.25, "cpu", False, 0.0)
        for name in ("Traffic Cone", "Barrier", "Tree"):
            self.assertIn(name, report)


class TestAutolabelFlow(unittest.TestCase):
    """编排层:注入假推理函数跑全流程(不碰 ultralytics)。"""

    @classmethod
    def setUpClass(cls):
        cls.registry = load_classes()

    def test_full_flow_with_fake_infer(self):
        with TemporaryDirectory() as d:
            event = Path(d)
            images = event / "images"
            images.mkdir()
            for i in range(1, 4):
                (images / f"{i:06d}.jpg").write_bytes(b"")

            def fake_infer(img, conf):
                # 第 1 帧:car + bicycle;第 2 帧:零检出;第 3 帧:person
                if img.stem == "000001":
                    return [_box(2, 0.9), _box(1, 0.8)]
                if img.stem == "000003":
                    return [_box(0, 0.6)]
                return []

            rc = al.autolabel(event, self.registry, infer_fn=fake_infer)
            self.assertEqual(rc, 0)

            labels = event / "labels"
            # txt 内容:car 1 框 / 空文件 / person 1 框
            self.assertEqual((labels / "000001.txt").read_text().count("\n") + 1, 1)
            self.assertEqual((labels / "000002.txt").read_text(), "")
            self.assertTrue((labels / "000003.txt").read_text().startswith("4 "))
            # classes.txt 9 类
            self.assertEqual(len((labels / "classes.txt").
                                 read_text().strip().splitlines()), 9)
            # 报告文件已写出且含零检出提醒
            report = (labels / "autolabel_report.txt").read_text(encoding="utf-8")
            self.assertIn("000002.jpg", report)

    def test_missing_images_dir_returns_1(self):
        with TemporaryDirectory() as d:
            self.assertEqual(
                al.autolabel(Path(d), self.registry, infer_fn=lambda i, c: []), 1)

    def test_bad_conf_returns_1(self):
        with TemporaryDirectory() as d:
            (Path(d) / "images").mkdir()
            self.assertEqual(al.autolabel(Path(d), self.registry, conf=1.5,
                                          infer_fn=lambda i, c: []), 1)

    def test_missing_ultralytics_friendly(self):
        """没有 ultralytics 的环境:infer_fn=None 且 import 失败 -> 退出码 1。"""
        import sys
        from unittest import mock
        with TemporaryDirectory() as d:
            images = Path(d) / "images"
            images.mkdir()
            (images / "000001.jpg").write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ultralytics": None}):
                self.assertEqual(al.autolabel(Path(d), self.registry), 1)


if __name__ == "__main__":
    unittest.main()

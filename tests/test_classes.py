# -*- coding: utf-8 -*-
"""classes.py 的单元测试(标准库 unittest)。

运行方式(仓库根目录下)::

    set PYTHONPATH=src
    python -m unittest discover tests
"""
import unittest

from traffic_dataset.classes import load_classes


class TestClassRegistry(unittest.TestCase):
    """类别别名解析、大小写不敏感、未知类别处理。"""

    @classmethod
    def setUpClass(cls):
        # 用仓库根目录 configs/classes.yaml(默认路径解析)
        cls.registry = load_classes()

    def test_nine_classes_defined(self):
        """必须恰好定义 9 类,id 为 0-8 连续。"""
        self.assertEqual(self.registry.num_classes, 9)
        self.assertEqual(sorted(self.registry.id_to_name), list(range(9)))

    def test_canonical_name_resolution(self):
        """规范名本身可以解析。"""
        self.assertEqual(self.registry.name_to_id("Car"), 0)
        self.assertEqual(self.registry.name_to_id("Traffic Cone"), 5)
        self.assertEqual(self.registry.name_to_id("Traffic light"), 8)

    def test_alias_resolution(self):
        """别名映射到正确 id。"""
        self.assertEqual(self.registry.name_to_id("motorbike"), 3)
        self.assertEqual(self.registry.name_to_id("person"), 4)
        self.assertEqual(self.registry.name_to_id("cone"), 5)
        self.assertEqual(self.registry.name_to_id("trafficlight"), 8)

    def test_case_insensitive(self):
        """大小写不敏感。"""
        self.assertEqual(self.registry.name_to_id("car"), 0)
        self.assertEqual(self.registry.name_to_id("CAR"), 0)
        self.assertEqual(self.registry.name_to_id("Car"), 0)
        self.assertEqual(self.registry.name_to_id("MOTORBIKE"), 3)
        self.assertEqual(self.registry.name_to_id("Person"), 4)
        self.assertEqual(self.registry.name_to_id("CONE"), 5)
        self.assertEqual(self.registry.name_to_id("Traffic Light"), 8)

    def test_whitespace_stripped(self):
        """名称两端空白不影响解析。"""
        self.assertEqual(self.registry.name_to_id("  car  "), 0)

    def test_unknown_returns_none(self):
        """不在 9 类里的类别返回 None(由调用方决定跳过)。"""
        self.assertIsNone(self.registry.name_to_id("bicycle"))
        self.assertIsNone(self.registry.name_to_id("airplane"))
        self.assertIsNone(self.registry.name_to_id("suitcase"))
        self.assertIsNone(self.registry.name_to_id(""))
        self.assertIsNone(self.registry.name_to_id(None))

    def test_id_to_name_roundtrip(self):
        """id -> 名称 -> id 往返一致。"""
        for cid in range(9):
            name = self.registry.id_to_name_str(cid)
            self.assertEqual(self.registry.name_to_id(name), cid)

    def test_names_order(self):
        """names 按 id 顺序,可直接写 YOLO classes.txt。"""
        self.assertEqual(
            self.registry.names,
            ["Car", "Bus", "Truck", "Motorcycle", "Pedestrian",
             "Traffic Cone", "Barrier", "Tree", "Traffic light"],
        )


if __name__ == "__main__":
    unittest.main()

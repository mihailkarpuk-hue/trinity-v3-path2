# -*- coding: utf-8 -*-
"""Приёмка Этапа 0.1 — канон 108 ключей."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)

from ядро.ключи_108 import KEYS_108, KEYS_BY_GROUP, KEYS_SHA256  # noqa: E402

КАТАЛОГ = os.path.join(КОРЕНЬ, "данные", "клеточки", "каталог.json")
ЗЕРКАЛО = os.path.join(КОРЕНЬ, "данные", "ключи_108.json")


class TestКлючи(unittest.TestCase):
    def test_сумма_групп_108(self):
        total = sum(len(v) for v in KEYS_BY_GROUP.values())
        self.assertEqual(total, 108)
        self.assertEqual(len(KEYS_108), 108)
        self.assertEqual(len(set(KEYS_108)), 108)

    def test_размеры_групп(self):
        expected = {
            "TEMPORAL": 16, "SPECTRAL": 18, "VOCAL": 8, "MUSICAL": 14,
            "SPATIAL": 5, "PERCEPTUAL": 7, "MOVEMENT": 8, "ADVANCED": 27, "AXES": 5,
        }
        for g, n in expected.items():
            self.assertEqual(len(KEYS_BY_GROUP[g]), n, g)

    def test_hash_стабилен(self):
        payload = "\n".join(KEYS_108).encode("utf-8")
        self.assertEqual(hashlib.sha256(payload).hexdigest(), KEYS_SHA256)
        зеркало = json.load(open(ЗЕРКАЛО, encoding="utf-8"))
        self.assertEqual(зеркало["sha256"], KEYS_SHA256)
        self.assertEqual(зеркало["ключи"], KEYS_108)

    def test_150_клеток_полны(self):
        d = json.load(open(КАТАЛОГ, encoding="utf-8"))
        клетки = next(v for v in d.values() if isinstance(v, list))
        self.assertEqual(len(клетки), 150)
        for c in клетки:
            p = c.get("параметры104") or {}
            for k in KEYS_108:
                self.assertIn(k, p, f"{c['id']}: нет ключа {k}")
                self.assertIsInstance(p[k], (int, float))


if __name__ == "__main__":
    unittest.main()

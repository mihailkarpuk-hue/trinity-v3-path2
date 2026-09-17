# -*- coding: utf-8 -*-
"""Приёмка строения атома: протон + электрон + образ_кусочек."""
from __future__ import annotations

import unittest

import numpy as np

from ядро.строение_атома import (
    КЛЮЧИ_ПРОТОН,
    КЛЮЧИ_ЭЛЕКТРОН,
    атом_полон,
    разрезать_108,
    собрать_образ_кусочек,
    собрать_слоты_атома,
)
from ядро.ключи_108 import KEYS_108


class TestСтроениеАтома(unittest.TestCase):
    def test_нарезка_108(self):
        self.assertEqual(len(КЛЮЧИ_ПРОТОН), 61)
        self.assertEqual(len(КЛЮЧИ_ЭЛЕКТРОН), 47)
        self.assertEqual(len(КЛЮЧИ_ПРОТОН) + len(КЛЮЧИ_ЭЛЕКТРОН), 108)
        self.assertEqual(set(КЛЮЧИ_ПРОТОН) | set(КЛЮЧИ_ЭЛЕКТРОН), set(KEYS_108))

    def test_образ_синхрон(self):
        img = собрать_образ_кусочек(
            t=0.5, старт=11025,
            пики=[{"частота": 440.0, "амплитуда": 0.3}],
            форма={"поз_x": 0.25, "размер": 0.3, "частота_дом": 440.0},
            палитра=[[200.0, 10, 20, 30], [800.0, 200, 100, 50]],
            источник="тест",
        )
        self.assertEqual(img["t"], 0.5)
        self.assertEqual(img["старт"], 11025)
        self.assertTrue(img["синхрон"])
        self.assertGreaterEqual(len(img["точки"]), 1)
        self.assertNotEqual(img["точки"][0]["r"], img["точки"][0]["b"])  # из палитры

    def test_слоты_и_полнота(self):
        params = {k: 0.1 for k in KEYS_108}
        слоты = собрать_слоты_атома(
            t=1.0, старт=100,
            params_108=params,
            пики=[[440, 0.2]],
            форма={"поз_x": 0.1, "размер": 0.2, "частота_дом": 440},
            палитра=[[440, 1, 2, 3]],
            источник="x",
        )
        atom = {
            "t": 1.0, "старт": 100,
            "протон": слоты["протон"],
            "электрон": слоты["электрон"],
            "образ_кусочек": слоты["образ_кусочек"],
        }
        self.assertTrue(атом_полон(atom))
        self.assertEqual(len(слоты["протон"]), 61)
        self.assertEqual(len(слоты["электрон"]), 47)

    def test_разрезать(self):
        p, e = разрезать_108({"spectral_centroid": 1.5, "bpm": 120.0})
        self.assertEqual(p["spectral_centroid"], 1.5)
        self.assertEqual(e["bpm"], 120.0)


if __name__ == "__main__":
    unittest.main()

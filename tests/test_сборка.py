# -*- coding: utf-8
"""Приёмка Этапа 5 — сборка из кирpичей по скелетам атомов."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from ядро.сборка import sha256_wav  # noqa: E402
from ядро.синтез import синтезировать  # noqa: E402
from ядро.сборка import SR
from ядро.словарь_кирпичей import загрузить, собрать_гены_клетки  # noqa: E402

ДЕМО = ("bazis_12_kaplya", "etalon_dozhd", "etalon_komar")


class TestСборка(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.словарь = загрузить()

    def test_клетки_разные(self):
        hs = []
        for cid in ДЕМО:
            gens, dur, _ = собрать_гены_клетки(cid, self.словарь)
            y = синтезировать(gens, sr=SR, dur=dur)
            hs.append(sha256_wav(y))
        self.assertEqual(len(set(hs)), len(ДЕМО))

    def test_komar_минимум_2_типа(self):
        _, _, состав = собрать_гены_клетки("etalon_komar", self.словарь)
        self.assertGreaterEqual(len(состав), 2)

    def test_детерминизм(self):
        gens1, dur1, _ = собрать_гены_клетки("etalon_dozhd", self.словарь)
        gens2, dur2, _ = собрать_гены_клетки("etalon_dozhd", self.словарь)
        y1 = синтезировать(gens1, sr=SR, dur=dur1)
        y2 = синтезировать(gens2, sr=SR, dur=dur2)
        np.testing.assert_array_equal(y1, y2)

    def test_живость_меняет_звук(self):
        gens0, dur0, _ = собрать_гены_клетки("etalon_dozhd", self.словарь, живость=0.0)
        gens1, dur1, _ = собрать_гены_клетки("etalon_dozhd", self.словарь, живость=1.0)
        y0 = синтезировать(gens0, sr=SR, dur=dur0)
        y1 = синтезировать(gens1, sr=SR, dur=dur1)
        n = min(len(y0), len(y1))
        diff = float(np.mean(np.abs(y0[:n] - y1[:n])))
        self.assertNotEqual(sha256_wav(y0), sha256_wav(y1))
        self.assertGreater(diff, 0.005)


if __name__ == "__main__":
    unittest.main()

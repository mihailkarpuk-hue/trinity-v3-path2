# -*- coding: utf-8 -*-
"""Приёмка Этапа 1 — атомизация V2 (треки) на синтетике."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)

from ядро.атомизация import атомизировать, атомизировать_синтетику  # noqa: E402
from ядро.паспорт_атома import ядро  # noqa: E402


class TestАтомизация(unittest.TestCase):
    def test_чирп_один_трек(self):
        x = атомизировать_синтетику("chirp", dur=0.2)
        atoms = атомизировать(x)
        tonal = [a for a in atoms if a.get("harmonicity", 0) > 0.5]
        self.assertGreaterEqual(len(tonal), 1, "нет тональных атомов")
        best = max(tonal, key=lambda a: a.get("lifetime", 0))
        y = ядро(best)
        self.assertAlmostEqual(y["freq_slope"], 5.0, delta=5.0 * 0.2)

    def test_тон_440_плоский(self):
        x = атомизировать_синтетику("tone", dur=0.3)
        atoms = атомизировать(x)
        slopes = [abs(ядро(a)["freq_slope"]) for a in atoms if a.get("harmonicity", 0) > 0.5]
        self.assertTrue(slopes, "нет тональных атомов")
        self.assertLess(max(slopes), 0.2)

    def test_клик_attack(self):
        x = атомизировать_синтетику("click", dur=0.05)
        atoms = атомизировать(x)
        self.assertGreater(len(atoms), 0)
        best = max(atoms, key=lambda a: a.get("amp", 0))
        self.assertLess(ядро(best)["attack_ratio"], 0.1)

    def test_поля_ядра_заполнены(self):
        x = атомизировать_синтетику("chirp")
        atoms = атомизировать(x)
        for a in atoms:
            self.assertIn("freq_slope", a)
            self.assertIn("attack_ratio", a)
            self.assertIn("decay_shape", a)
            self.assertIn("freq_t", a)
            self.assertIn("amp_t", a)
            self.assertLessEqual(len(a["freq_t"]), 32)


if __name__ == "__main__":
    unittest.main()

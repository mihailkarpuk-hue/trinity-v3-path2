# -*- coding: utf-8
"""Приёмка Этапа 2 — ГЕН и round-trip синтез."""
from __future__ import annotations

import hashlib
import os
import sys
import unittest

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from ядро.ген import валиден, ген_из_атома, гены_из_атомов, определить_режим_синтеза  # noqa: E402
from ядро.синтез import синтезировать  # noqa: E402


class TestGen(unittest.TestCase):
    def test_ген_из_атома_поля(self):
        atom = {
            "birth": 0.1,
            "lifetime": 0.05,
            "freq": 440,
            "freq_slope": 0.5,
            "harmonicity": 0.8,
            "amp": 0.3,
            "attack_ratio": 0.2,
            "decay_shape": -3,
            "phase": "тон",
            "freq_t": [400, 500],
            "amp_t": [0.1, 0.5, 0.2],
        }
        g = ген_из_атома(atom)
        self.assertEqual(g["фаза"], "тон")
        self.assertAlmostEqual(g["freq"], 440)
        self.assertTrue(валиден(g))

    def test_валиден_отклоняет(self):
        self.assertFalse(валиден({"фаза": "???", "lifetime": 1, "freq": 440, "amp": 0.5}))
        self.assertFalse(валиден({"фаза": "тон", "lifetime": 0, "freq": 440, "amp": 0.5}))

    def test_синтез_детерминизм(self):
        gens = гены_из_атомов([
            {"birth": 0, "lifetime": 0.08, "freq": 880, "harmonicity": 0.9, "amp": 0.4, "phase": "тон"},
            {"birth": 0.05, "lifetime": 0.06, "freq": 1200, "harmonicity": 0.2, "amp": 0.2, "phase": "шум"},
        ])
        a = синтезировать(gens, sr=22050, dur=0.2)
        b = синтезировать(gens, sr=22050, dur=0.2)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.max(np.abs(a)), 0.01)

    def test_режимы_синтеза(self):
        meta_am = {"параметры104": {"harmonic_ratio": 1.0}}
        g_am = [{
            "фаза": "тон", "birth": 0, "lifetime": 1.0, "freq": 220, "amp": 1.0,
            "harmonicity": 1.0, "amp_t": [0.5, 1.0, 0.5, 1.0] * 8, "freq_t": [220] * 32,
        }]
        self.assertEqual(определить_режим_синтеза(g_am, meta_am), "am")
        meta_fm = {"параметры104": {"harmonic_ratio": 0.4}}
        g_fm = [{
            "фаза": "тон", "birth": 0, "lifetime": 1.0, "freq": 220, "amp": 1.0,
            "harmonicity": 1.0, "amp_t": [1.0] * 32,
            "freq_t": [200, 240, 200, 240] * 8,
        }]
        self.assertEqual(определить_режим_синтеза(g_fm, meta_fm), "fm")
        meta_saw = {"параметры104": {"harmonic_ratio": 0.999}}
        g_saw = [
            {"фаза": "шум", "freq": 100 * (i + 1), "amp": 0.1, "lifetime": 1.0, "birth": 0}
            for i in range(12)
        ]
        self.assertEqual(определить_режим_синтеза(g_saw, meta_saw), "saw")


if __name__ == "__main__":
    unittest.main()

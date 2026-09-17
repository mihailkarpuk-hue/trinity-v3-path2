# -*- coding: utf-8
"""Приёмка Этапа 3 — словарь кирпичей."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from ядро.ген import ген_из_атома  # noqa: E402
from ядро.словарь_кирпичей import (  # noqa: E402
    SEED,
    NULL_MARGIN,
    k_means,
    вектор_типа_из_гена,
    загрузить_атомы_корпуса,
    нулевой_тест,
    перемешать_оси,
    сэмпл_гена,
    zscore,
)


class TestСловарь(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Xn, cls.mu, cls.sd, cls.meta, cls.atoms = загрузить_атомы_корпуса()

    def test_корпус_не_пуст(self):
        self.assertGreater(len(self.Xn), 500)

    def test_вектор_типа_12_осей(self):
        g = ген_из_атома({"freq": 440, "lifetime": 0.05, "harmonicity": 0.8, "amp": 0.3})
        v = вектор_типа_из_гена(g)
        self.assertEqual(len(v), 12)

    def test_перемешивание_детерминировано(self):
        a = перемешать_оси(self.Xn, SEED)
        b = перемешать_оси(self.Xn, SEED)
        np.testing.assert_array_equal(a, b)

    def test_kmeans_детерминирован(self):
        l1, c1 = k_means(self.Xn, 20, seed=SEED)
        l2, c2 = k_means(self.Xn, 20, seed=SEED)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_almost_equal(c1, c2)

    def test_нулевой_тест_структура(self):
        r = нулевой_тест(self.Xn)
        self.assertIn("пройден", r)
        self.assertIn("silhouette_реальные", r)
        self.assertGreater(len(r["silhouette_реальные"]), 5)

    def test_сэмпл_детерминирован(self):
        тип = {
            "id": 0,
            "центроид": {k: 0.0 for k in (
                "фаза_тон", "фаза_шум", "фаза_переход", "log_lifetime",
                "freq_slope", "harmonicity", "attack_ratio", "decay_shape",
                "noise_ratio", "band_width_rel", "mod_depth", "mod_rate",
            )},
            "sigma": {k: 0.01 for k in (
                "фаза_тон", "фаза_шум", "фаза_переход", "log_lifetime",
                "freq_slope", "harmonicity", "attack_ratio", "decay_shape",
                "noise_ratio", "band_width_rel", "mod_depth", "mod_rate",
            )},
        }
        тип["центроид"]["фаза_тон"] = 1.0
        тип["центроид"]["log_lifetime"] = -2.0
        тип["центроид"]["harmonicity"] = 0.7
        atom = self.atoms[0]
        mu = self.mu
        sd = self.sd
        g1 = сэмпл_гена(тип, atom, mu, sd)
        g2 = сэмпл_гена(тип, atom, mu, sd)
        self.assertEqual(g1["birth"], g2["birth"])
        self.assertAlmostEqual(g1["harmonicity"], g2["harmonicity"], places=9)


if __name__ == "__main__":
    unittest.main()

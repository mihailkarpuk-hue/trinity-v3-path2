# -*- coding: utf-8 -*-
"""Приёмка Этапа 0.2 — метрика d₉."""
from __future__ import annotations

import os
import sys
import unittest

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)

from ядро.метрика import d9, загрузить_индекс, сбросить_кэш  # noqa: E402


class TestМетрика(unittest.TestCase):
    def setUp(self):
        сбросить_кэш()

    def tearDown(self):
        сбросить_кэш()

    def test_d9_aa_zero(self):
        idx = загрузить_индекс()
        c = idx["клетки"][0]["вектор"]
        self.assertAlmostEqual(d9(c, c, mu=idx["mu"], sd=idx["sd"]), 0.0, places=9)

    def test_d9_симметрия(self):
        idx = загрузить_индекс()
        a = idx["клетки"][0]["вектор"]
        b = idx["клетки"][1]["вектор"]
        mu, sd = idx["mu"], idx["sd"]
        self.assertAlmostEqual(d9(a, b, mu=mu, sd=sd), d9(b, a, mu=mu, sd=sd), places=9)

    def test_смена_весов_меняет_d9(self):
        idx = загрузить_индекс()
        a = idx["клетки"][0]["вектор"]
        b = idx["клетки"][10]["вектор"]
        mu, sd = idx["mu"], idx["sd"]
        d0 = d9(a, b, mu=mu, sd=sd)
        d1 = d9(a, b, mu=mu, sd=sd, веса={"TEMPORAL": 10.0})
        self.assertNotAlmostEqual(d1, d0, places=6)
        self.assertGreater(d1, d0)


if __name__ == "__main__":
    unittest.main()

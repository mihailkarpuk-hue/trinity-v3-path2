# -*- coding: utf-8 -*-
"""Приёмка пути 2: кирпич-момент (corr, не d₉)."""
from __future__ import annotations

import unittest

import numpy as np

from ядро.кирпич_момент import SR, в_кирпичи, из_кирпичей


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    da, db = float(np.dot(a, a)), float(np.dot(b, b))
    if da < 1e-18 or db < 1e-18:
        return 0.0
    return float(np.dot(a, b) / np.sqrt(da * db))


class TestКирпичМомент(unittest.TestCase):
    def test_синус_roundtrip(self):
        t = np.arange(int(0.4 * SR)) / SR
        x = np.sin(2 * np.pi * 440.0 * t)
        y = из_кирпичей(в_кирпичи(x, SR), длина=len(x), sr=SR)
        self.assertGreaterEqual(_corr(x, y), 0.999)

    def test_детерминизм(self):
        t = np.arange(int(0.2 * SR)) / SR
        x = 0.5 * np.sin(2 * np.pi * 220.0 * t)
        b = в_кирпичи(x, SR)
        y1 = из_кирпичей(b, длина=len(x), sr=SR)
        y2 = из_кирпичей(b, длина=len(x), sr=SR)
        self.assertTrue(np.array_equal(y1, y2))


if __name__ == "__main__":
    unittest.main()

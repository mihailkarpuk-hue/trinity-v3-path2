# -*- coding: utf-8 -*-
"""Приёмка сборки организмов (путь 2)."""
from __future__ import annotations

import hashlib
import unittest

import numpy as np

from ядро.организм import РЕЦЕПТЫ, загрузить_манифест, собрать_по_имени


class TestОрганизм(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.man = загрузить_манифест()

    def test_рецепты_есть(self):
        for имя in ("шторм", "костёр", "вьюга", "пульс", "вздох_а"):
            self.assertIn(имя, РЕЦЕПТЫ)

    def test_шторм_детерминизм(self):
        a = собрать_по_имени("шторм", органы=self.man)
        b = собрать_по_имени("шторм", органы=self.man)
        self.assertTrue(np.array_equal(a, b))
        self.assertGreater(len(a), 1000)

    def test_все_рецепты_собираются(self):
        for имя in РЕЦЕПТЫ:
            y = собрать_по_имени(имя, органы=self.man)
            self.assertGreater(np.max(np.abs(y)), 0.1, msg=имя)
            h1 = hashlib.sha256(y.tobytes()).hexdigest()
            h2 = hashlib.sha256(
                собрать_по_имени(имя, органы=self.man).tobytes()
            ).hexdigest()
            self.assertEqual(h1, h2, msg=имя)


if __name__ == "__main__":
    unittest.main()

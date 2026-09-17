# -*- coding: utf-8
"""Приёмка Этапа 4 — обратная проекция образ → скелеты."""
from __future__ import annotations

import json
import os
import sys
import unittest

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from ядро.образ_в_атомы import (  # noqa: E402
    T_ОБРАЗА_ПО_УМОЛЧ,
    атомы_в_облако,
    из_клетки,
    облако_в_скелеты,
    совпадают,
)

КЛЕТКИ = os.path.join(КОРЕНЬ, "данные", "клеточки")
TOL = 0.02

ТЕСТ_КЛЕТКИ = ("bazis_12_kaplya", "etalon_dozhd", "живая_А")


class TestОбратнаяПроекция(unittest.TestCase):
    def test_kaplya_roundtrip(self):
        atoms, облако, meta = из_клетки("bazis_12_kaplya")
        sk = облако_в_скелеты(облако, meta=meta, ограничить_плотность=False)
        ok, worst = совпадают(atoms, sk, tol=TOL)
        self.assertTrue(ok, f"worst rel err {worst:.4f}")

    def test_несколько_клеток(self):
        for cid in ТЕСТ_КЛЕТКИ:
            with self.subTest(cid=cid):
                atoms, облако, meta = из_клетки(cid)
                sk = облако_в_скелеты(облако, meta=meta, ограничить_плотность=False)
                ok, worst = совпадают(atoms, sk, tol=TOL)
                self.assertTrue(ok, f"{cid}: worst={worst:.4f}")

    def test_канон_roundtrip(self):
        atoms, _, _ = из_клетки("bazis_12_kaplya")
        облако, meta = атомы_в_облако(atoms, канон=True, t_образа=T_ОБРАЗА_ПО_УМОЛЧ)
        sk = облако_в_скелеты(облако, meta=meta, ограничить_плотность=False)
        self.assertEqual(len(sk), len(atoms))

    def test_скелеты_имеют_кривизну(self):
        atoms, облако, meta = из_клетки("bazis_12_kaplya")
        sk = облако_в_скелеты(облако, meta=meta)
        for s in sk:
            self.assertIn("лок_кривизна", s)
            self.assertIn("лок_плотность", s)

    def test_плотность_ограничение(self):
        """>24 точек в 50 мс → сжатие."""
        pts = [{"x": 0, "y": 0, "z": 0.5, "size": 1}] * 30
        meta = {"режим": "канон", "t_образа": T_ОБРАЗА_ПО_УМОЛЧ}
        sk = облако_в_скелеты(pts, meta=meta, ограничить_плотность=True)
        self.assertLessEqual(len(sk), 25)


if __name__ == "__main__":
    unittest.main()

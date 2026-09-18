# -*- coding: utf-8 -*-
"""Запас каталога, когда данные/клеточки/ нет в git."""
import json
import os
import unittest

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ИНДЕКС = os.path.join(КОРЕНЬ, "данные", "узнавание_индекс_108.json")
ПОЛКА = os.path.join(КОРЕНЬ, "данные", "библиотека_природы", "манифест.json")


class TestКаталогЗапас(unittest.TestCase):
    def test_индекс_150_и_дождь(self):
        with open(ИНДЕКС, encoding="utf-8") as f:
            индекс = json.load(f)
        клетки = индекс["клетки"]
        self.assertEqual(len(клетки), 150)
        ids = {к["id"] for к in клетки}
        self.assertIn("etalon_dozhd", ids)
        дождь = next(к for к in клетки if к["id"] == "etalon_dozhd")
        self.assertEqual(дождь["группа"], "базис")

    def test_полка_природы_на_диске(self):
        with open(ПОЛКА, encoding="utf-8") as f:
            полка = json.load(f)
        органы = {о["стихия_id"]: о for о in полка["органы"]}
        self.assertEqual(len(органы), 7)
        self.assertIn("dozhd", органы)
        for стихия, о in органы.items():
            путь = os.path.join(КОРЕНЬ, "данные", "библиотека_природы", о["файл"])
            self.assertTrue(os.path.isfile(путь), путь)
            self.assertEqual(о["файл"], f"{стихия}.json")


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8
"""Построить данные/геометрия_в_кирпич.json из словаря типов."""
from __future__ import annotations

import json
import os
import sys

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from ядро.словарь_кирпичей import загрузить  # noqa: E402

ВЫХОД = os.path.join(КОРЕНЬ, "данные", "геометрия_в_кирпич.json")


def _pool(типы, *keys):
    ids = []
    for t in типы:
        имя = t.get("имя") or ""
        if any(k in имя for k in keys):
            ids.append(int(t["id"]))
    return sorted(set(ids))[:12]


def построить() -> dict:
    сл = загрузить()
    типы = сл.get("типы") or []
    return {
        "версия": 1,
        "описание": "назначение типа скелету по локальной геометрии (данные, не код)",
        "пороги": {
            "кривизна_низкая": 0.15,
            "плотность_высокая": 2.0,
        },
        "pools": {
            "тональные": _pool(типы, "тональная", "чирп"),
            "шумовые": _pool(типы, "шорох", "шипящее", "шумовое"),
            "ударные": _pool(типы, "удар", "пульс"),
        },
        "hue": {
            "тёплый": {"pref_noise": True, "pref_tonal": False},
            "холодный": {"pref_tonal": True, "pref_noise": False},
        },
        "живость_по_умолчанию": 0.7,
        "t_образа_по_умолчанию": 3.0,
    }


if __name__ == "__main__":
    d = построить()
    with open(ВЫХОД, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"✓ {ВЫХОД} · pools: { {k: len(v) for k, v in d['pools'].items()} }")

# -*- coding: utf-8 -*-
"""Построение узнавание_индекс_108.json из каталога клеточек.

Каждая клетка → вектор из 108 параметров (параметры104-агрегат),
z-нормировка (mu/sd по всем клеткам). Совместимо с узнавание.js (d₉ kNN).

usage: python3 построить_индекс_108.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from ядро.ключи_108 import KEYS_108, KEYS_SHA256

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
КАТАЛОГ = os.path.join(КОРЕНЬ, "данные", "клеточки", "каталог.json")
ВЫХОД = os.path.join(КОРЕНЬ, "данные", "узнавание_индекс_108.json")


def _вектор(п: dict, ключи: list[str]) -> list[float] | None:
    if not п or len(п) < 100:
        return None
    v = []
    for k in ключи:
        x = п.get(k)
        try:
            v.append(float(x))
        except (TypeError, ValueError):
            v.append(0.0)
    return v


def построить() -> None:
    ключи = list(KEYS_108)
    assert len(ключи) == 108, f"ожидалось 108 ключей, есть {len(ключи)}"

    d = json.load(open(КАТАЛОГ, encoding="utf-8"))
    список_ключ = next(k for k, v in d.items() if isinstance(v, list))
    клетки_вход = d[список_ключ]

    строки = []
    матрица = []
    for c in клетки_вход:
        п = c.get("параметры104") or {}
        v = _вектор(п, ключи)
        if v is None:
            continue
        строки.append({
            "id": c.get("id"),
            "название": c.get("название") or c.get("имя") or c.get("id"),
            "группа": c.get("группа") or "",
        })
        матрица.append(v)

    M = np.asarray(матрица, dtype=np.float64)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd[sd < 1e-9] = 1.0  # защита от деления на ноль (константные оси)

    клетки = []
    for мета, вектор in zip(строки, матрица):
        клетки.append({**мета, "вектор": [round(float(x), 6) for x in вектор]})

    индекс = {
        "версия": 2,
        "ключи_sha256": KEYS_SHA256,
        "оси": ключи,
        "mu": [round(float(x), 6) for x in mu],
        "sd": [round(float(x), 6) for x in sd],
        "клетки": клетки,
    }
    with open(ВЫХОД, "w", encoding="utf-8") as f:
        json.dump(индекс, f, ensure_ascii=False)
    print(f"✓ {os.path.basename(ВЫХОД)}: {len(клетки)} клеток · 108 осей · sha256={KEYS_SHA256[:16]}…")


if __name__ == "__main__":
    построить()

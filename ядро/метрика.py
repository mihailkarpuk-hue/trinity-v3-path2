# -*- coding: utf-8 -*-
"""Блочная метрика d₉ — единый судья расстояний в 108-мерном пространстве.

d₉(a,b) = Σ_g w_g · d_g(a,b) / √|g|

d_g — евклид по z-нормированным осям группы g (μ/σ из индекса).
w_g — из данные/веса_групп.json (по умолчанию все = 1).
"""
from __future__ import annotations

import json
import math
import os
from typing import Mapping, Sequence

from ядро.ключи_108 import KEYS_BY_GROUP, KEYS_108

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ВЕСА_ПУТЬ = os.path.join(КОРЕНЬ, "данные", "веса_групп.json")
ИНДЕКС_ПУТЬ = os.path.join(КОРЕНЬ, "данные", "узнавание_индекс_108.json")

_кэш_индекс: dict | None = None
_кэш_веса: dict[str, float] | None = None


def _загрузить_веса() -> dict[str, float]:
    global _кэш_веса
    if _кэш_веса is not None:
        return _кэш_веса
    if os.path.isfile(ВЕСА_ПУТЬ):
        d = json.load(open(ВЕСА_ПУТЬ, encoding="utf-8"))
        _кэш_веса = {g: float(d.get(g, 1.0)) for g in KEYS_BY_GROUP}
    else:
        _кэш_веса = {g: 1.0 for g in KEYS_BY_GROUP}
    return _кэш_веса


def загрузить_индекс(путь: str | None = None) -> dict:
    global _кэш_индекс
    if _кэш_индекс is not None and путь is None:
        return _кэш_индекс
    p = путь or ИНДЕКС_ПУТЬ
    idx = json.load(open(p, encoding="utf-8"))
    if путь is None:
        _кэш_индекс = idx
    return idx


def сбросить_кэш() -> None:
    global _кэш_индекс, _кэш_веса
    _кэш_индекс = None
    _кэш_веса = None


def _индексы_групп() -> dict[str, list[int]]:
    pos = {k: i for i, k in enumerate(KEYS_108)}
    return {g: [pos[k] for k in keys if k in pos] for g, keys in KEYS_BY_GROUP.items()}


def вектор_из_словаря(п: Mapping[str, float]) -> list[float]:
    return [float(п.get(k) or 0.0) for k in KEYS_108]


def d_g(
    a: Sequence[float],
    b: Sequence[float],
    индексы: list[int],
    mu: Sequence[float],
    sd: Sequence[float],
) -> float:
    s = 0.0
    for i in индексы:
        za = (a[i] - mu[i]) / sd[i]
        zb = (b[i] - mu[i]) / sd[i]
        d = za - zb
        s += d * d
    return math.sqrt(s)


def d9(
    a: Sequence[float],
    b: Sequence[float],
    *,
    mu: Sequence[float] | None = None,
    sd: Sequence[float] | None = None,
    веса: Mapping[str, float] | None = None,
) -> float:
    """Блочное расстояние d₉ между двумя 108-векторами."""
    if mu is None or sd is None:
        idx = загрузить_индекс()
        mu = idx["mu"]
        sd = idx["sd"]
    w = dict(_загрузить_веса())
    if веса:
        w.update({k: float(v) for k, v in веса.items()})
    groups = _индексы_групп()
    total = 0.0
    for g, inds in groups.items():
        if not inds:
            continue
        dg = d_g(a, b, inds, mu, sd)
        total += w.get(g, 1.0) * dg / math.sqrt(len(inds))
    return total


def d9_по_группам(
    a: Sequence[float],
    b: Sequence[float],
    *,
    mu: Sequence[float] | None = None,
    sd: Sequence[float] | None = None,
) -> dict[str, float]:
    """d_g для каждой из 9 групп (без весов, для отчётов)."""
    if mu is None or sd is None:
        idx = загрузить_индекс()
        mu = idx["mu"]
        sd = idx["sd"]
    groups = _индексы_групп()
    return {
        g: d_g(a, b, inds, mu, sd) / math.sqrt(len(inds)) if inds else 0.0
        for g, inds in groups.items()
    }

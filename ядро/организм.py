# -*- coding: utf-8 -*-
"""Сборка ОРГАНИЗМА из органов библиотеки моментов (путь 2).

Иерархия ПРАВИЛО_СБОРКИ.md: атом → клетка → орган → организм.
Организм = сумма органов во времени (тип ≠ место: сдвиг старта — параметр сборки).

Ничего не нормируем в параметрах органов. Пик волны нормализуем только
на выходе wav (чтобы не клиппить при сложении) — это амплитуда носителя, не фенотип.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ядро.кирпич_момент import ОКНО, SR, из_кирпичей

_КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
БИБ_ПО_УМОЛЧАНИЮ = os.path.join(_КОРЕНЬ, "данные", "библиотека_кирпичей")


@dataclass(frozen=True)
class СлойОргана:
    """Один орган в организме: источник + место во времени + громкость."""

    источник: str
    сдвиг_с: float = 0.0
    громкость: float = 1.0
    лимит_атомов: int | None = None  # None = все; иначе первые N (как раскат грома)


def загрузить_манифест(путь_биб: str = БИБ_ПО_УМОЛЧАНИЮ) -> dict[str, dict]:
    with open(os.path.join(путь_биб, "манифест.json"), encoding="utf-8") as f:
        M = json.load(f)
    return {o["источник"]: o for o in M["органы"]}


def атомы_органа(
    cid: str,
    органы: dict[str, dict],
    *,
    сдвиг_с: float = 0.0,
    лимит: int | None = None,
    путь_биб: str = БИБ_ПО_УМОЛЧАНИЮ,
) -> list[dict]:
    """Атомы органа: спектр из npz, старт из манифеста + сдвиг (тип ≠ место)."""
    if cid not in органы:
        raise KeyError(f"орган не в библиотеке: {cid}")
    z = np.load(os.path.join(путь_биб, cid + ".npz"))
    mags, phs = z["mag"], z["phase"]
    атомы = органы[cid]["атомы"]
    if лимит is not None:
        атомы = атомы[:лимит]
    ds = int(сдвиг_с * SR)
    return [
        {
            "старт": int(a["старт"]) + ds,
            "спектр_магнитуда": mags[j],
            "спектр_фаза": phs[j],
        }
        for j, a in enumerate(атомы)
    ]


def собрать_организм(
    слои: Iterable[СлойОргана],
    *,
    органы: dict[str, dict] | None = None,
    путь_биб: str = БИБ_ПО_УМОЛЧАНИЮ,
    нормализовать_пик: bool = True,
) -> np.ndarray:
    """Слои органов → одна волна (сумма). Детерминированно."""
    слои = list(слои)
    if not слои:
        return np.zeros(1, dtype=np.float64)
    if органы is None:
        органы = загрузить_манифест(путь_биб)

    партии: list[tuple[np.ndarray, float]] = []
    nmax = 0
    for слой in слои:
        bricks = атомы_органа(
            слой.источник, органы,
            сдвиг_с=слой.сдвиг_с,
            лимит=слой.лимит_атомов,
            путь_биб=путь_биб,
        )
        if not bricks:
            continue
        n = max(b["старт"] for b in bricks) + ОКНО + 10
        nmax = max(nmax, n)
        y = из_кирпичей(bricks, длина=n, sr=SR)
        партии.append((y, float(слой.громкость)))

    if not партии or nmax <= 0:
        return np.zeros(1, dtype=np.float64)

    s = np.zeros(nmax, dtype=np.float64)
    for y, g in партии:
        if len(y) < nmax:
            y = np.pad(y, (0, nmax - len(y)))
        s += y[:nmax] * g

    if нормализовать_пик:
        peak = float(np.max(np.abs(s))) + 1e-12
        s = s / peak
    return s


# --- рецепты организмов (звук, которого не было в корпусе как целого) ---

РЕЦЕПТЫ: dict[str, list[СлойОргана]] = {
    # канон: гром + дождь
    "шторм": [
        СлойОргана("etalon_dozhd", 0.0, 0.8),
        СлойОргана("etalon_grom", 1.2, 1.0, лимит_атомов=24),
    ],
    # огонь + ветер
    "костёр": [
        СлойОргана("etalon_ogon", 0.0, 1.0),
        СлойОргана("etalon_veter", 0.4, 0.55),
    ],
    # ветер + дождь (без грома)
    "вьюга": [
        СлойОргана("etalon_veter", 0.0, 0.9),
        СлойОргана("etalon_dozhd", 0.6, 0.7),
        СлойОргана("etalon_pesok", 1.0, 0.35),
    ],
    # тело: сердце + дыхание
    "пульс": [
        СлойОргана("etalon_serdce", 0.0, 1.0),
        СлойОргана("etalon_dyhanie", 0.8, 0.65),
    ],
    # природа + буква (организм пересекает классы корпуса)
    "вздох_а": [
        СлойОргана("etalon_dyhanie", 0.0, 0.85),
        СлойОргана("живая_А", 1.5, 1.0),
    ],
}


def собрать_по_имени(имя: str, **kw) -> np.ndarray:
    if имя not in РЕЦЕПТЫ:
        raise KeyError(f"нет рецепта «{имя}»; есть: {sorted(РЕЦЕПТЫ)}")
    return собрать_организм(РЕЦЕПТЫ[имя], **kw)

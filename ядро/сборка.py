# -*- coding: utf-8
"""Сборка звука из скелетов + словарь кирпичей.

Только словарь — без доступа к природным записям каталога.
Скелет (birth, freq, amp) + тип кирпича → ГЕН → синтез.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np

from ядро.ген import валиден
from ядро.синтез import синтезировать
from ядро.словарь_кирпичей import (
    SEED,
    _seed_atom,
    ген_из_вектора_типа,
    загрузить as загрузить_словарь,
    ТИП_КЛЮЧИ,
)

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ТАБЛИЦА_ПУТЬ = os.path.join(КОРЕНЬ, "данные", "геометрия_в_кирпич.json")
ЖИВОСТЬ_ПО_УМОЛЧ = 0.7
SR = 22050


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def загрузить_таблицу() -> dict:
    return json.load(open(ТАБЛИЦА_ПУТЬ, encoding="utf-8"))


def _hue_тепло(скелет: Mapping) -> str:
    c = скелет.get("цвет") or {}
    r = float(c.get("r") or 128)
    g = float(c.get("g") or 128)
    b = float(c.get("b") or 128)
    if r + g + b < 1:
        r, g, b = r * 255, g * 255, b * 255
    return "тёплый" if r > b else "холодный"


def _score_типа(скелет: Mapping, тип: dict, таблица: dict) -> float:
    пороги = таблица.get("пороги") or {}
    k_low = float(пороги.get("кривизна_низкая", 0.15))
    p_high = float(пороги.get("плотность_высокая", 2.0))
    имя = str(тип.get("имя") or "")
    крив = float(скелет.get("лок_кривизна") or 0)
    плот = float(скелет.get("лок_плотность") or 1)
    s = float(тип.get("доля") or 0) * 0.5

    pools = таблица.get("pools") or {}
    if крив <= k_low:
        if any(x in имя for x in ("тональная", "чирп")):
            s += 3.0
        if тип["id"] in pools.get("тональные", []):
            s += 1.5
    else:
        if any(x in имя for x in ("шорох", "шипящее", "шумовое")):
            s += 3.0
        if тип["id"] in pools.get("шумовые", []):
            s += 1.5

    if плот >= p_high:
        if any(x in имя for x in ("удар", "пульс")):
            s += 2.5
        if тип["id"] in pools.get("ударные", []):
            s += 1.5

    hue = _hue_тепло(скелет)
    pref = (таблица.get("hue") or {}).get(hue, {})
    if pref.get("pref_noise") and "шум" in имя:
        s += 1.0
    if pref.get("pref_tonal") and "тональная" in имя:
        s += 1.0

    c = тип.get("центроид") or {}
    tid = int(тип["id"])
    freq = float(скелет.get("freq") or 440)
    birth = float(скелет.get("birth") or 0)
    target_h = max(0.05, min(0.95, 1.0 - math.log10(max(freq, 60)) / math.log10(9000)))
    th = float(c.get("harmonicity") or 0.5)
    s += max(0.0, 2.5 - abs(th - target_h) * 5)
    if freq > 800 and крив <= k_low and ("тональная" in имя or "чирп" in имя):
        s += 0.8
    if freq < 400 and "шорох" in имя:
        s += 0.6
    if birth > 1.5 and "шорох" in имя:
        s += 0.5
    if birth < 0.5 and ("удар" in имя or "чирп" in имя):
        s += 0.5
    s += (hash((round(birth, 3), round(freq, 1), tid)) % 1000) * 1e-4
    return s


def назначить_тип(
    скелет: Mapping,
    словарь: dict,
    таблица: dict | None = None,
) -> dict:
    """Скелет → лучший тип кирпича по таблице геометрии."""
    таблица = таблица or загрузить_таблицу()
    типы = словарь.get("типы") or []
    if not типы:
        raise ValueError("пустой словарь кирпичей")
    best = max(типы, key=lambda t: _score_типа(скелет, t, таблица))
    return best


def сэмпл_гена(
    скелет: Mapping,
    тип: dict,
    словарь: dict,
    *,
    живость: float = ЖИВОСТЬ_ПО_УМОЛЧ,
) -> dict:
    """Скелет + распределение типа → полный ГЕН (детерминированно)."""
    mu = np.asarray(словарь["mu"], dtype=np.float64)
    sd = np.asarray(словарь["sd"], dtype=np.float64)
    birth = float(скелет["birth"])
    freq = float(скелет["freq"])
    amp = float(скелет["amp"])
    tid = int(тип["id"])
    rng = _rng(_seed_atom(birth, freq, tid))
    c = np.array([тип["центроид"][k] for k in ТИП_КЛЮЧИ], dtype=np.float64)
    sig = np.array([тип["sigma"][k] for k in ТИП_КЛЮЧИ], dtype=np.float64) * float(живость)
    raw = c + rng.normal(size=len(c)) * sig
    g = ген_из_вектора_типа(raw, birth=birth, freq=freq, amp=amp)
    if not валиден(g):
        g["amp"] = max(0.01, min(1.0, amp))
    return g


def состав_из_назначений(назначения: list[dict]) -> dict[str, float]:
    """Доли типов в сборке (для плашки UI)."""
    counts: dict[int, int] = {}
    for n in назначения:
        tid = int(n["тип"]["id"])
        counts[tid] = counts.get(tid, 0) + 1
    total = len(назначения) or 1
    names = {int(t["id"]): t["имя"] for t in (назначения[0]["словарь"]["типы"] if назначения else [])}
    return {names.get(k, str(k)): round(v / total, 3) for k, v in sorted(counts.items())}


def собрать_гены(
    скелеты: Sequence[Mapping],
    *,
    словарь: dict | None = None,
    таблица: dict | None = None,
    живость: float = ЖИВОСТЬ_ПО_УМОЛЧ,
) -> tuple[list[dict], list[dict], dict[str, float]]:
    """Скелеты → гены + метаданные назначений + состав."""
    словарь = словарь or загрузить_словарь()
    таблица = таблица or загрузить_таблицу()
    назначения = []
    гены = []
    for sk in скелеты:
        typ = назначить_тип(sk, словарь, таблица)
        g = сэмпл_гена(sk, typ, словарь, живость=живость)
        гены.append(g)
        назначения.append({"скелет": dict(sk), "тип": typ, "словарь": словарь})
    # состав по именам
    counts: dict[str, int] = {}
    for n in назначения:
        name = n["тип"]["имя"]
        counts[name] = counts.get(name, 0) + 1
    total = len(назначения) or 1
    состав = {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda x: -x[1])}
    return гены, назначения, состав


def собрать_wav(
    скелеты: Sequence[Mapping],
    *,
    словарь: dict | None = None,
    таблица: dict | None = None,
    живость: float = ЖИВОСТЬ_ПО_УМОЛЧ,
    sr: int = SR,
    dur: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Полная сборка → (waveform, состав)."""
    гены, _, состав = собрать_гены(скелеты, словарь=словарь, таблица=таблица, живость=живость)
    if dur is None and гены:
        dur = max(g["birth"] + g["lifetime"] for g in гены) + 0.05
    y = синтезировать(гены, sr=sr, dur=dur)
    return y, состав


def sha256_wav(y: np.ndarray) -> str:
    return hashlib.sha256(y.astype(np.float64).tobytes()).hexdigest()

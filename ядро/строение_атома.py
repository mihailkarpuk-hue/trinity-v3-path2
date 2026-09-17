# -*- coding: utf-8 -*-
"""Строение атома-момента: ПРОТОН + ЭЛЕКТРОН + ЧАСТИЧКА ЖИВОГО ОБРАЗА.

Канон: СТРОЕНИЕ_АТОМА_МОМЕНТА.md
  АТОМ = протон (параметры + спектр-носитель) + электрон (параметры)
       + образ_кусочек (синхронно со звуком, тот же t/старт).

Нарезка 108 (KEYS_BY_GROUP):
  ПРОТОН  (61): SPECTRAL + VOCAL + MUSICAL + PERCEPTUAL + mfcc_0..12 + zcr
  ЭЛЕКТРОН (47): TEMPORAL + SPATIAL + MOVEMENT + mfcc_delta_* + AXES
"""
from __future__ import annotations

import math
from typing import Any

from ядро.ключи_108 import KEYS_108, KEYS_BY_GROUP

# --- нарезка протон / электрон ---
_ПРОТОН_ГРУППЫ = ("SPECTRAL", "VOCAL", "MUSICAL", "PERCEPTUAL")
_ЭЛЕКТРОН_ГРУППЫ = ("TEMPORAL", "SPATIAL", "MOVEMENT", "AXES")

КЛЮЧИ_ПРОТОН: tuple[str, ...] = tuple(
    k for g in _ПРОТОН_ГРУППЫ for k in KEYS_BY_GROUP[g]
) + tuple(f"mfcc_{i}" for i in range(13)) + ("zero_crossing_rate",)

КЛЮЧИ_ЭЛЕКТРОН: tuple[str, ...] = tuple(
    k for g in _ЭЛЕКТРОН_ГРУППЫ for k in KEYS_BY_GROUP[g]
) + tuple(f"mfcc_delta_{i}" for i in range(13))

assert len(КЛЮЧИ_ПРОТОН) + len(КЛЮЧИ_ЭЛЕКТРОН) == 108, (
    len(КЛЮЧИ_ПРОТОН), len(КЛЮЧИ_ЭЛЕКТРОН)
)
assert set(КЛЮЧИ_ПРОТОН) | set(КЛЮЧИ_ЭЛЕКТРОН) == set(KEYS_108)
assert set(КЛЮЧИ_ПРОТОН).isdisjoint(КЛЮЧИ_ЭЛЕКТРОН)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x or abs(x) == float("inf"):  # NaN/Inf
            return default
        return x
    except (TypeError, ValueError):
        return default


def разрезать_108(params: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Полный dict → (протон, электрон). Недостающие ключи = 0 (кроме spatial — см. статус)."""
    протон = {k: round(_f(params.get(k)), 6) for k in КЛЮЧИ_ПРОТОН}
    электрон = {k: round(_f(params.get(k)), 6) for k in КЛЮЧИ_ЭЛЕКТРОН}
    return протон, электрон


def цвет_из_палитры(freq_hz: float, палитра: list) -> tuple[int, int, int]:
    """Интерполяция RGB по частоте из палитры живого образа. НЕ freqToHue."""
    if not палитра:
        return 128, 128, 128
    f = max(20.0, float(freq_hz))
    # палитра: [freq, r, g, b] sorted
    if f <= палитра[0][0]:
        _, r, g, b = палитра[0]
        return int(r), int(g), int(b)
    if f >= палитра[-1][0]:
        _, r, g, b = палитра[-1]
        return int(r), int(g), int(b)
    for i in range(len(палитра) - 1):
        f0, r0, g0, b0 = палитра[i]
        f1, r1, g1, b1 = палитра[i + 1]
        if f0 <= f <= f1:
            t = 0.0 if f1 <= f0 else (f - f0) / (f1 - f0)
            return (
                int(r0 + t * (r1 - r0)),
                int(g0 + t * (g1 - g0)),
                int(b0 + t * (b1 - b0)),
            )
    _, r, g, b = палитра[-1]
    return int(r), int(g), int(b)


def _y_log(freq: float) -> float:
    """Y образа: log-частота, нормировка грубо к [0..1] для 60..9000 Гц."""
    f = max(60.0, min(float(freq), 9000.0))
    return round(
        (math.log(f) - math.log(60.0)) / (math.log(9000.0) - math.log(60.0)), 4
    )


def собрать_образ_кусочек(
    *,
    t: float,
    старт: int,
    пики: list,
    форма: dict,
    палитра: list,
    источник: str,
) -> dict:
    """Частичка живого образа — синхронно со звуком (тот же t/старт)."""
    точки = []
    # пики: либо dict{частота,амплитуда}, либо [freq, amp]
    raw = пики or []
    for p in raw[:12]:
        if isinstance(p, dict):
            freq, amp = float(p.get("частота") or 0), float(p.get("амплитуда") or 0)
        else:
            freq, amp = float(p[0]), float(p[1]) if len(p) > 1 else 0.0
        if freq <= 0:
            continue
        r, g, b = цвет_из_палитры(freq, палитра)
        точки.append({
            "x": round(float(форма.get("поз_x") or 0), 4),
            "y": _y_log(freq),
            "частота": round(freq, 1),
            "размер": round(max(amp, 0.0), 4),
            "r": r, "g": g, "b": b,
        })
    # если пиков нет — одна точка на доминанте (всё равно слот образа есть)
    if not точки:
        fd = float(форма.get("частота_дом") or 200.0)
        r, g, b = цвет_из_палитры(fd, палитра)
        точки.append({
            "x": round(float(форма.get("поз_x") or 0), 4),
            "y": _y_log(fd),
            "частота": round(fd, 1),
            "размер": round(float(форма.get("размер") or 0), 4),
            "r": r, "g": g, "b": b,
        })
    return {
        "t": round(float(t), 4),
        "старт": int(старт),
        "синхрон": True,
        "источник": источник,
        "цвет_из": "палитра_живого_образа" if палитра else "нейтраль",
        "точки": точки,
    }


def mfcc_дельты_по_ряду(ряд_параметров: list[dict]) -> list[dict]:
    """mfcc_delta_i для каждого атома из соседа слева (первый = 0)."""
    out = []
    prev = None
    for p in ряд_параметров:
        d = {}
        for i in range(13):
            k = f"mfcc_{i}"
            cur = _f(p.get(k))
            d[f"mfcc_delta_{i}"] = round(cur - (_f(prev.get(k)) if prev else cur), 6)
        out.append(d)
        prev = p
    return out


def обогатить_окно_108(seg, sr: int) -> dict:
    """103 из analyze_full_103 + 5 AXES-заглушек (оси подставит орган)."""
    from atoms_full103 import analyze_full_103  # noqa: WPS433
    import numpy as np
    seg = np.asarray(seg, dtype=np.float64)
    # analyze ждёт достаточную длину — паддинг не врёт спектру окна целиком
    if len(seg) < sr // 20:
        seg = np.pad(seg, (0, max(0, sr // 20 - len(seg))))
    p = analyze_full_103(seg, sr)
    for k in KEYS_BY_GROUP["AXES"]:
        p.setdefault(k, 0.0)
    return p


def собрать_слоты_атома(
    *,
    t: float,
    старт: int,
    params_108: dict,
    пики: list,
    форма: dict,
    палитра: list,
    источник: str,
    mono: bool = True,
) -> dict:
    """Возвращает {протон, электрон, образ_кусочек, полнота}."""
    протон, электрон = разрезать_108(params_108)
    spatial_статус = "н/п_моно" if mono else "ok"
    образ = собрать_образ_кусочек(
        t=t, старт=старт, пики=пики, форма=форма,
        палитра=палитра, источник=источник,
    )
    n_p = sum(1 for k, v in протон.items() if v != 0.0 or k in params_108)
    # считаем заполненность: ключ есть в исходном params
    n_p_fill = sum(1 for k in КЛЮЧИ_ПРОТОН if k in params_108)
    n_e_fill = sum(1 for k in КЛЮЧИ_ЭЛЕКТРОН if k in params_108)
    if mono:
        # spatial не считаем «живыми»
        n_e_live = sum(
            1 for k in КЛЮЧИ_ЭЛЕКТРОН
            if k in params_108 and k not in KEYS_BY_GROUP["SPATIAL"]
        )
        n_e_ожид = len(КЛЮЧИ_ЭЛЕКТРОН) - len(KEYS_BY_GROUP["SPATIAL"])
    else:
        n_e_live, n_e_ожид = n_e_fill, len(КЛЮЧИ_ЭЛЕКТРОН)
    return {
        "протон": протон,
        "электрон": электрон,
        "образ_кусочек": образ,
        "spatial_статус": spatial_статус,
        "полнота": {
            "протон": f"{n_p_fill}/{len(КЛЮЧИ_ПРОТОН)}",
            "электрон_без_spatial" if mono else "электрон": f"{n_e_live}/{n_e_ожид}",
            "образ_точек": len(образ["точки"]),
            "образ_синхрон": True,
        },
    }


def атом_полон(атом: dict) -> bool:
    """Приёмка: есть протон, электрон, образ_кусочек с тем же t/старт."""
    if not атом.get("протон") or not атом.get("электрон"):
        return False
    img = атом.get("образ_кусочек") or {}
    if not img.get("точки"):
        return False
    if not img.get("синхрон"):
        return False
    # t в манифесте округляется до 4 знаков — сравниваем на той же сетке
    if round(float(img.get("t", -1)), 4) != round(float(атом.get("t", -2)), 4):
        return False
    if int(img.get("старт", -1)) != int(атом.get("старт", -2)):
        return False
    return True

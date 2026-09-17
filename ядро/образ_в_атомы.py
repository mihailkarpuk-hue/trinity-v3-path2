# -*- coding: utf-8
"""Обратная проекция: облако точек образа → скелеты атомов.

Инверсия проекция_образ.js (режим «физика»):
  X = время, Y = log10(freq), Z = amp.

Канон (ТЗ §4.2):
  birth = norm(x) · T_образа
  freq  = exp(norm(y) · [ln 60 .. ln 9000])
  amp   = norm(z) → [0.05 .. 1.0]
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np

from ядро.пороги import BAND_F_HI, BAND_F_LO

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
КЛЕТКИ = os.path.join(КОРЕНЬ, "данные", "клеточки")

# Масштабы сцены — зеркало проекция_образ.js (режим физика)
X_SCALE = 1.8
Y_SCALE = 2.4
Z_SCALE = 1.2
T_ОБРАЗА_ПО_УМОЛЧ = 3.0
AMP_MIN, AMP_MAX = 0.05, 1.0
ОКНО_BIRTH = 0.05
MAX_СКЕЛЕТОВ_В_ОКНЕ = 24
K_СОСЕДЕЙ = 8


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _norm(v: float, a: float, b: float) -> float:
    if abs(b - a) < 1e-12:
        return 0.5
    return (v - a) / (b - a)


def _границы_атомов(atoms: Sequence[Mapping]) -> dict[str, float]:
    births = [float(a.get("birth") or 0) for a in atoms]
    lfs = [math.log10(max(1.0, float(a.get("freq") or 20))) for a in atoms]
    amps = [float(a.get("amp") or 0) for a in atoms]
    return {
        "tmin": min(births),
        "tmax": max(births),
        "fmin": min(lfs),
        "fmax": max(lfs),
        "amax": max(max(amps), 1e-6),
    }


def атомы_в_облако(
    atoms: Sequence[Mapping],
    *,
    t_образа: float | None = None,
    канон: bool = False,
) -> tuple[list[dict], dict]:
    """Прямая проекция атомов → облако (для тестов round-trip)."""
    if not atoms:
        return [], {}
    b = _границы_атомов(atoms)
    if канон:
        meta = {"режим": "канон", "t_образа": t_образа or T_ОБРАЗА_ПО_УМОЛЧ}
    else:
        meta = {**b, "режим": "физика", "t_образа": t_образа or T_ОБРАЗА_ПО_УМОЛЧ}
    облако = []
    for a in atoms:
        birth = float(a.get("birth") or 0)
        freq = max(1.0, float(a.get("freq") or 20))
        amp = float(a.get("amp") or 0)
        if канон:
            nx = _clip01(birth / meta["t_образа"])
            x = (2 * nx - 1) * X_SCALE
            ny = _clip01((math.log(freq) - math.log(BAND_F_LO)) / (math.log(BAND_F_HI) - math.log(BAND_F_LO)))
            y = (ny - 0.5) * Y_SCALE
            nz = _clip01((amp - AMP_MIN) / (AMP_MAX - AMP_MIN))
            z = nz * Z_SCALE
        else:
            x = (2 * _norm(birth, b["tmin"], b["tmax"]) - 1) * X_SCALE
            y = (_norm(math.log10(freq), b["fmin"], b["fmax"]) - 0.5) * Y_SCALE
            z = (amp / b["amax"]) * Z_SCALE
        pt = {
            "x": round(x, 6),
            "y": round(y, 6),
            "z": round(z, 6),
            "size": float(a.get("size") or max(0.05, amp)),
        }
        if a.get("color_r") is not None:
            pt["r"] = int(a["color_r"])
            pt["g"] = int(a.get("color_g") or 0)
            pt["b"] = int(a.get("color_b") or 0)
        облако.append(pt)
    return облако, meta


def _точка_в_скелет(
    pt: Mapping,
    meta: Mapping,
) -> dict[str, float]:
    x, y, z = float(pt["x"]), float(pt["y"]), float(pt["z"])
    w = float(pt.get("size") or 1.0)

    if meta.get("режим") == "физика" and "tmin" in meta:
        birth = (x / X_SCALE + 1) / 2
        birth = birth * (float(meta["tmax"]) - float(meta["tmin"])) + float(meta["tmin"])
        logf = y / Y_SCALE + 0.5
        logf = logf * (float(meta["fmax"]) - float(meta["fmin"])) + float(meta["fmin"])
        freq = 10.0 ** logf
        amp = (z / Z_SCALE) * float(meta["amax"])
        amp = max(0.0, amp)
    else:
        t = float(meta.get("t_образа") or T_ОБРАЗА_ПО_УМОЛЧ)
        nx = _clip01((x / X_SCALE + 1) / 2)
        birth = nx * t
        ny = _clip01(y / Y_SCALE + 0.5)
        ln_lo, ln_hi = math.log(BAND_F_LO), math.log(BAND_F_HI)
        freq = math.exp(ny * (ln_hi - ln_lo) + ln_lo)
        nz = _clip01(z / Z_SCALE)
        amp = AMP_MIN + nz * (AMP_MAX - AMP_MIN)

    amp_out = amp if meta.get("режим") == "физика" else max(AMP_MIN, min(AMP_MAX, amp))
    sk = {
        "birth": round(max(0.0, birth), 6),
        "freq": round(max(BAND_F_LO, min(BAND_F_HI, freq)), 4),
        "amp": round(amp_out, 6),
        "_w": w,
    }
    if "r" in pt:
        sk["цвет"] = {"r": pt["r"], "g": pt.get("g"), "b": pt.get("b")}
    return sk


def _слить_окно(group: list[dict]) -> dict:
    wsum = sum(g["_w"] for g in group) or 1.0
    out = {
        "birth": sum(g["birth"] * g["_w"] for g in group) / wsum,
        "freq": sum(g["freq"] * g["_w"] for g in group) / wsum,
        "amp": sum(g["amp"] * g["_w"] for g in group) / wsum,
    }
    if group[0].get("цвет"):
        out["цвет"] = group[0]["цвет"]
    return out


def _ограничить_плотность(скелеты: list[dict]) -> list[dict]:
    """≤ MAX_СКЕЛЕТОВ_В_ОКНЕ на каждые ОКНО_BIRTH секунд."""
    if not скелеты:
        return []
    sorted_s = sorted(скелеты, key=lambda s: s["birth"])
    out: list[dict] = []
    i = 0
    n = len(sorted_s)
    while i < n:
        t0 = sorted_s[i]["birth"]
        group = []
        while i < n and sorted_s[i]["birth"] - t0 <= ОКНО_BIRTH + 1e-9:
            group.append(sorted_s[i])
            i += 1
        if len(group) <= MAX_СКЕЛЕТОВ_В_ОКНЕ:
            out.extend(group)
        else:
            group.sort(key=lambda g: g["_w"], reverse=True)
            keep = group[:MAX_СКЕЛЕТОВ_В_ОКНЕ]
            merged_rest = [_слить_окно(group[MAX_СКЕЛЕТОВ_В_ОКНЕ:])] if len(group) > MAX_СКЕЛЕТОВ_В_ОКНЕ else []
            out.extend(keep + merged_rest)
    return out


def _лок_кривизна(pts: np.ndarray, i: int, k: int = K_СОСЕДЕЙ) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    d = np.linalg.norm(pts - pts[i], axis=1)
    d[i] = np.inf
    nn = np.argsort(d)[: min(k, n - 1)]
    block = pts[nn]
    if len(block) < 2:
        return 0.0
    c = block - block.mean(axis=0)
    cov = c.T @ c / max(len(block) - 1, 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(np.maximum(evals, 0))[::-1]
    s = float(evals.sum())
    return float(evals[-1] / s) if s > 1e-12 else 0.0


def _лок_плотность(pts: np.ndarray, i: int, k: int = K_СОСЕДЕЙ) -> float:
    n = len(pts)
    if n <= 1:
        return 1.0
    d = np.linalg.norm(pts - pts[i], axis=1)
    d[i] = np.inf
    nn = np.sort(d)[: min(k, n - 1)]
    r = float(np.mean(nn)) if len(nn) else 1.0
    return round(1.0 / max(r, 1e-3), 4)


def облако_в_скелеты(
    облако: Sequence[Mapping],
    *,
    meta: Mapping | None = None,
    t_образа: float = T_ОБРАЗА_ПО_УМОЛЧ,
    ограничить_плотность: bool = True,
) -> list[dict]:
    """Облако → скелеты {birth, freq, amp, лок_кривизна, лок_плотность, цвет?}."""
    if not облако:
        return []
    m = dict(meta or {})
    if "режим" not in m:
        m.update({"режим": "канон", "t_образа": t_образа})

    raw = [_точка_в_скелет(p, m) for p in облако]
    if ограничить_плотность:
        raw = _ограничить_плотность(raw)

    pts = np.array([[s["birth"], math.log(max(s["freq"], 1)), s["amp"]] for s in raw], dtype=np.float64)
    out = []
    for i, s in enumerate(raw):
        sk = {
            "birth": s["birth"],
            "freq": s["freq"],
            "amp": s["amp"],
            "лок_кривизна": round(_лок_кривизна(pts, i), 6),
            "лок_плотность": _лок_плотность(pts, i),
        }
        if s.get("цвет"):
            sk["цвет"] = s["цвет"]
        out.append(sk)
    return out


def образ_в_атомы(
    облако: Sequence[Mapping],
    *,
    meta: Mapping | None = None,
    t_образа: float = T_ОБРАЗА_ПО_УМОЛЧ,
) -> list[dict]:
    """Публичный API — то же, что облако_в_скелеты."""
    return облако_в_скелеты(облако, meta=meta, t_образа=t_образа)


def из_json_облака(path: str) -> list[dict]:
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, list):
        return list(d)
    return list(d.get("точки") or d.get("points") or [])


def из_obj(path: str) -> list[dict]:
    """Минимальный OBJ → облако {x,y,z}."""
    verts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append({"x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3]), "size": 1.0})
    return verts


def из_клетки(cell_id: str) -> tuple[list[dict], list[dict], dict]:
    """Клетка → (исходные атомы, облако, meta) для round-trip."""
    cat = json.load(open(os.path.join(КЛЕТКИ, "каталог.json"), encoding="utf-8"))
    cells = next(v for v in cat.values() if isinstance(v, list))
    c = next(x for x in cells if x["id"] == cell_id)
    jp = os.path.join(КЛЕТКИ, c.get("атомы") or "")
    rec = json.load(open(jp, encoding="utf-8"))
    atoms = rec.get("atoms") or []
    облако, meta = атомы_в_облако(atoms, канон=False)
    return atoms, облако, meta


def совпадают(
    orig: Sequence[Mapping],
    recon: Sequence[Mapping],
    *,
    tol: float = 0.02,
) -> tuple[bool, float]:
    """Сравнение birth/freq/amp (допуск tol относительный)."""
    if len(orig) != len(recon):
        return False, math.inf

    def key(a):
        return (round(float(a.get("birth") or 0), 4), round(float(a.get("freq") or 0), 1))

    o = sorted(orig, key=key)
    r = sorted(recon, key=key)

    errs = []
    for a, b in zip(o, r):
        for field in ("birth", "freq", "amp"):
            va = float(a.get(field) or 0)
            vb = float(b.get(field) or 0)
            denom = max(abs(va), abs(vb), 1e-9)
            errs.append(abs(va - vb) / denom)
    worst = max(errs) if errs else 0.0
    return worst <= tol, worst

# -*- coding: utf-8
"""ГЕН — порождающие параметры атома (обратимый синтез).

ФЕНОТИП = params_104 (108 осей) — только измерение; обратимость через ГЕН.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ядро.фаза import фаза

ГЕН_КЛЮЧИ = (
    "фаза", "birth", "lifetime", "freq", "freq_slope",
    "harmonic_index", "harmonicity", "amp", "attack_ratio", "decay_shape",
    "noise_ratio", "band_center", "band_width", "mod_depth", "mod_rate",
    "freq_t", "amp_t", "режим",
)

РЕЖИМЫ = ("grain", "am", "fm", "saw")


def _fase_str(atom: dict) -> str:
    ph = atom.get("phase")
    if isinstance(ph, str) and ph in ("тон", "шум", "переход"):
        return ph
    h = float(atom.get("harmonicity") or 0)
    return фаза(h)


def _band_from_freq(freq: float) -> tuple[float, float]:
    f = max(60.0, float(freq or 440))
    w = max(f * 0.25, 80.0)
    return f, min(w, f * 0.9)


def _mod_from_amp_t(amp_t: list[float], lifetime: float) -> tuple[float, float]:
    if len(amp_t) < 4 or lifetime <= 0:
        return 0.0, 0.0
    a = np.asarray(amp_t, dtype=np.float64)
    mx, mn, mu = float(a.max()), float(a.min()), float(a.mean())
    if mu < 1e-9 or mx <= mn:
        return 0.0, 0.0
    depth = (mx - mn) / (mx + mn)
    detrend = a - mu
    if len(detrend) >= 8:
        spec = np.abs(np.fft.rfft(detrend))
        freqs = np.fft.rfftfreq(len(detrend), d=lifetime / max(len(detrend) - 1, 1))
        spec[0] = 0.0
        if spec.max() > 1e-12:
            k = int(np.argmax(spec[1:])) + 1
            return min(1.0, depth * 1.05), max(0.5, float(freqs[k]))
    return min(1.0, depth), 0.0


def _mod_from_freq_t(freq_t: list[float], lifetime: float) -> tuple[float, float]:
    if len(freq_t) < 4 or lifetime <= 0:
        return 0.0, 0.0
    f = np.asarray(freq_t, dtype=np.float64)
    mu = float(f.mean())
    if mu < 1e-9:
        return 0.0, 0.0
    depth = (float(f.max()) - float(f.min())) / (2.0 * mu)
    detrend = f - mu
    if len(detrend) >= 8:
        spec = np.abs(np.fft.rfft(detrend))
        freqs = np.fft.rfftfreq(len(detrend), d=lifetime / max(len(detrend) - 1, 1))
        spec[0] = 0.0
        if spec.max() > 1e-12:
            k = int(np.argmax(spec[1:])) + 1
            return min(1.0, depth * 1.2), max(0.5, float(freqs[k]))
    return min(1.0, depth), 0.0


def _amp_t_oscillates(amp_t: list[float]) -> bool:
    if len(amp_t) < 4:
        return False
    a = np.asarray(amp_t, dtype=np.float64)
    mu = float(a.mean())
    return mu > 1e-9 and float(a.std()) / mu > 0.05


def определить_режим_синтеза(gens: list[dict], meta: dict | None = None) -> str:
    """V4: режим синтеза по ГЕНам и параметрам104 клетки."""
    if not gens:
        return "grain"
    p104 = (meta or {}).get("параметры104") or (meta or {}).get("parent_params_full") or {}
    hr = float(p104.get("harmonic_ratio") or 0.0)

    if len(gens) == 1:
        g = gens[0]
        amp_t = g.get("amp_t") or []
        freq_t = g.get("freq_t") or []
        if g.get("фаза") == "тон" and len(freq_t) >= 4:
            f = np.asarray(freq_t, dtype=np.float64)
            mu = float(f.mean())
            if mu > 1e-9 and float(f.std()) / mu > 0.025:
                if not _amp_t_oscillates(amp_t):
                    return "fm"
        if g.get("фаза") == "тон" and _amp_t_oscillates(amp_t):
            f_flat = len(freq_t) >= 2 and float(np.std(freq_t)) < 1.0
            if f_flat:
                return "am"

    if hr > 0.995 and len(gens) >= 12:
        f0 = min(float(g["freq"]) for g in gens)
        if f0 >= 30:
            hits = sum(
                1 for g in gens
                if abs(float(g["freq"]) / f0 - round(float(g["freq"]) / f0))
                / max(round(float(g["freq"]) / f0), 1) < 0.08
            )
            if hits >= max(8, int(len(gens) * 0.4)):
                return "saw"
    return "grain"


def ген_из_атома(atom: dict) -> dict[str, Any]:
    """Атом (трек) → ГЕН для синтеза."""
    h = float(atom.get("harmonicity") or 0)
    freq = float(atom.get("freq") or 440)
    bc, bw = _band_from_freq(freq)
    p104 = atom.get("params_104") or {}
    amp_t = list(atom.get("amp_t") or [])
    freq_t = list(atom.get("freq_t") or [])
    lifetime = float(atom.get("lifetime") or 0.05)
    mod_depth = float(p104.get("mod_depth") or atom.get("mod_depth") or 0.0)
    mod_rate = float(p104.get("mod_rate") or atom.get("mod_rate") or 0.0)
    if mod_depth <= 0.01 and len(amp_t) >= 4 and lifetime >= 0.35 and h > 0.15:
        md, mr = _mod_from_amp_t(amp_t, lifetime)
        if md > mod_depth:
            mod_depth, mod_rate = md, mr or mod_rate
    if mod_depth <= 0.01 and len(freq_t) >= 4 and lifetime >= 0.35 and h > 0.15:
        md, mr = _mod_from_freq_t(freq_t, lifetime)
        if md > mod_depth:
            mod_depth = md
            mod_rate = mod_rate or mr
    return {
        "фаза": _fase_str(atom),
        "birth": float(atom.get("birth") or 0),
        "lifetime": lifetime,
        "freq": freq,
        "freq_slope": float(atom.get("freq_slope") or 0),
        "harmonic_index": int(atom.get("harmonic_index") or (1 if h > 0.5 else 0)),
        "harmonicity": h,
        "amp": float(atom.get("amp") or 0.1),
        "attack_ratio": float(atom.get("attack_ratio") or 0.1),
        "decay_shape": float(atom.get("decay_shape") or -2.0),
        "noise_ratio": float(max(0.0, min(1.0, 1.0 - h))),
        "band_center": float(atom.get("band_center") or bc),
        "band_width": float(atom.get("band_width") or bw),
        "mod_depth": mod_depth,
        "mod_rate": mod_rate,
        "freq_t": freq_t,
        "amp_t": amp_t,
        "режим": "grain",
    }


def гены_из_атомов(atoms: list[dict]) -> list[dict]:
    return [ген_из_атома(a) for a in atoms]


def валиден(ген: Mapping[str, Any]) -> bool:
    try:
        if ген.get("фаза") not in ("тон", "шум", "переход"):
            return False
        if float(ген.get("lifetime") or 0) <= 0:
            return False
        if float(ген.get("freq") or 0) <= 0:
            return False
        if not (0 <= float(ген.get("amp") or 0) <= 1.5):
            return False
        ft = ген.get("freq_t") or []
        at = ген.get("amp_t") or []
        if len(ft) > 32 or len(at) > 32:
            return False
        rm = ген.get("режим")
        if rm is not None and rm not in РЕЖИМЫ:
            return False
        return True
    except (TypeError, ValueError):
        return False

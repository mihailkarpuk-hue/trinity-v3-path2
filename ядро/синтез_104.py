# -*- coding: utf-8 -*-
"""синтез_104.py — generative синтез из params_104 каждого атома.

FULL-путь (синтез.py) не заменяет — opt-in v2:
  TEMPORAL → ADSR grain
  VOCAL    → jitter / shimmer / breathiness
  оси      → per-band LFO + связь полос по локальному nestedness
  кресты   → фазы (условно: только при tonality > 0.35)

Использование:
  from синтез_104 import синтез_104_из_атомов
  y = синтез_104_из_атомов(atoms, crosses, meta=rec)
"""
from __future__ import annotations

import math

import numpy as np

from маппинг_104 import GrainSpec, adsr_envelope, операторы_гранулы
from паспорт_атома import has_full, полоса_индекс
from синтез import (
    NEST_BANDS,
    SR_DEFAULT,
    _phases_from_crosses,
    _prepare,
    связать_полосы,
)

HARM_PHASE_THRESH = 0.35
DEFAULT_GRAIN = 0.08


def _render_grain(spec: GrainSpec, sr: int, phase: float, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """Одна гранула → (start_sample, mono segment)."""
    gl = max(32, int(max(spec.lifetime, DEFAULT_GRAIN * 0.5) * sr))
    f = spec.freq
    if f <= 20 or f >= sr / 2 or spec.amp <= 0:
        return 0, np.zeros(0, dtype=np.float64)
    df = f * spec.jitter * rng.normal()
    da = 1.0 + spec.shimmer * rng.normal()
    f_use = max(20.0, f + df)
    amp = min(1.0, spec.amp * 6 * da)
    env = adsr_envelope(gl, sr, spec)
    w = 2 * math.pi * f_use / sr
    idx = np.arange(gl, dtype=np.float64)
    ton = spec.tonality * (1 - spec.noise_mix)
    sig = amp * 0.26 * env * np.sin(w * idx + phase)
    if spec.noise_mix > 0.05:
        sig += amp * 0.18 * spec.noise_mix * env * rng.normal(size=gl) * 0.5
    sig *= (1 - spec.breathiness * 0.15)
    return int(spec.birth * sr), sig.astype(np.float64)


def _band_lfo(n: int, sr: int, depth: float, rate: float, phase0: float = 0.0) -> np.ndarray:
    if depth <= 0.01 or rate <= 0:
        return np.ones(n, dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / sr
    return np.clip(1.0 + depth * np.sin(2 * math.pi * rate * t + phase0), 0.06, None)


def _couple_bands(band_signals: list[np.ndarray], nest_weights: list[float], sr: int) -> np.ndarray:
    """Связать полосы: common shape из взвешенного nestedness атомов полосы."""
    if not band_signals:
        return np.zeros(1, dtype=np.float64)
    n = max(len(b) for b in band_signals)
    out = np.zeros(n, dtype=np.float64)
    envs = []
    for b in band_signals:
        seg = np.zeros(n, dtype=np.float64)
        seg[: len(b)] = b
        from scipy import signal as sig
        e = np.abs(sig.hilbert(seg))
        if len(e) > 205:
            e = sig.savgol_filter(e, 201, 2)
        envs.append(np.maximum(e, 1e-9))
    if len(envs) < 2:
        for b in band_signals:
            out[: len(b)] += b
        return out
    rel = np.mean([e / (e.mean() + 1e-9) for e in envs], axis=0)
    alphas = [min(0.75, max(0.0, w * 0.85)) for w in nest_weights]
    a_mean = float(np.mean(alphas)) if alphas else 0.0
    if a_mean <= 0.02:
        for b in band_signals:
            out[: len(b)] += b
        return out
    for b, e, a in zip(band_signals, envs, alphas):
        seg = np.zeros(n, dtype=np.float64)
        seg[: len(b)] = b
        target = (1 - a) * e + a * (rel * (e.mean() + 1e-9))
        gain = np.clip(target / e, 0.25, 4.0)
        out += seg * gain
    return out


def синтез_104_из_атомов(
    atoms: list[dict],
    crosses: list[dict] | None = None,
    *,
    sr: int = SR_DEFAULT,
    dur: float | None = None,
    meta: dict | None = None,
    max_atoms: int = 8000,
    post_couple: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generative 104-path. Атомы без params_104 всё равно синтезируются (defaults)."""
    if not atoms:
        return np.zeros(sr // 10, dtype=np.float64)

    rng = rng or np.random.default_rng(0)
    list_a, crosses_a = _prepare(atoms, crosses, max_atoms)

    if dur is None:
        dur = max(float(a.get("birth") or 0) for a in list_a) + DEFAULT_GRAIN + 0.15
    n_total = int(dur * sr) + 2048

    specs: list[GrainSpec] = [операторы_гранулы(a, default_grain=DEFAULT_GRAIN) for a in list_a]

    phases = _phases_from_crosses(list_a, crosses_a)
    band_bufs: list[list[tuple[int, np.ndarray]]] = [[] for _ in NEST_BANDS]
    band_nest: list[list[float]] = [[] for _ in NEST_BANDS]

    for i, (a, spec) in enumerate(zip(list_a, specs)):
        harm = float(a.get("harmonicity") or 0)
        use_cross_phase = harm >= HARM_PHASE_THRESH
        ph = phases[i] if use_cross_phase else phases[i] * 0.4 + rng.uniform(0, 2 * math.pi) * 0.6
        if harm >= HARM_PHASE_THRESH and a.get("phase") is not None:
            ph = float(a["phase"])
        s0, grain = _render_grain(spec, sr, ph, rng)
        if len(grain) == 0:
            continue
        bi = min(spec.band, len(NEST_BANDS) - 1)
        lfo = _band_lfo(len(grain), sr, spec.mod_depth, spec.mod_rate, ph * 0.1)
        grain = grain * lfo
        band_bufs[bi].append((s0, grain))
        band_nest[bi].append(max(0.0, spec.nestedness))

    band_out: list[np.ndarray] = []
    nest_w: list[float] = []
    for bi, items in enumerate(band_bufs):
        if not items:
            continue
        seg_len = min(n_total, max(s0 + len(g) for s0, g in items))
        acc = np.zeros(seg_len, dtype=np.float64)
        for s0, g in items:
            e = min(len(g), seg_len - s0)
            if e > 0:
                acc[s0:s0 + e] += g[:e]
        band_out.append(acc)
        nest_w.append(float(np.mean(band_nest[bi])) if band_nest[bi] else 0.0)

    if len(band_out) >= 2:
        y = _couple_bands(band_out, nest_w, sr)
    else:
        y = np.zeros(n_total, dtype=np.float64)
        for items in band_bufs:
            for s0, g in items:
                e = min(len(g), n_total - s0)
                if e > 0:
                    y[s0:s0 + e] += g[:e]

    if len(y) < n_total:
        y = np.pad(y, (0, n_total - len(y)))
    else:
        y = y[:n_total]

    if post_couple and meta:
        p104 = meta.get("параметры104") or meta.get("parent_params_full") or {}
        target = float(p104.get("nestedness") or 0.0)
        if target > 0.05:
            y = связать_полосы(y, sr, target * 0.5)

    m = np.max(np.abs(y))
    return y / m * 0.9 if m > 0 else y


def синтез_104_из_json(rec: dict, *, sr: int = SR_DEFAULT, post_couple: bool = False) -> tuple[np.ndarray, int]:
    atoms = rec.get("atoms") or []
    crosses = rec.get("crosses")
    dur = rec.get("длительность_сек")
    y = синтез_104_из_атомов(
        atoms, crosses, sr=sr, dur=float(dur) + 0.1 if dur else None,
        meta=rec, post_couple=post_couple,
    )
    return y, sr


def статистика_готовности(atoms: list[dict]) -> dict:
    n = len(atoms)
    full = sum(1 for a in atoms if has_full(a))
    return {"atoms": n, "with_params_104": full, "pct": round(100 * full / n, 1) if n else 0}

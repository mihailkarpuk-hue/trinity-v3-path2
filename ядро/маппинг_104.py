# -*- coding: utf-8 -*-
"""Маппинг params_104 → операторы гранулы (GrainSpec).

Не 104 ручки — 8 групп → ~20 активных полей в MVP.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from паспорт_атома import get, оси, полоса_индекс, ядро


@dataclass
class GrainSpec:
    """Параметры одной гранулы из полного паспорта атома."""
    birth: float
    freq: float
    amp: float
    phase: float
    lifetime: float
    band: int
    # TEMPORAL → форма огибающей
    attack_s: float
    decay_s: float
    sustain: float
    # VOCAL → жизнь
    jitter: float      # относительная девиация freq
    shimmer: float     # относительная девиация amp
    breathiness: float
    # SPECTRAL / MUSICAL
    tonality: float
    noise_mix: float
    # оси (локальные)
    mod_depth: float
    mod_rate: float
    nestedness: float
    fd: float
    # MOVEMENT
    flow: float


@dataclass
class GenGrainSpec:
    """Гранула из ГЕН (Этап 2 round-trip)."""
    birth: float
    lifetime: float
    freq: float
    freq_slope: float
    amp: float
    фаза: str
    harmonic_index: int
    harmonicity: float
    attack_ratio: float
    decay_shape: float
    noise_ratio: float
    band_center: float
    band_width: float
    mod_depth: float
    mod_rate: float
    freq_t: list[float] = field(default_factory=list)
    amp_t: list[float] = field(default_factory=list)
    seed: int = 0


def _seed_гена(ген: dict) -> int:
    payload = f"{ген.get('band_center')}:{ген.get('birth')}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def grain_из_гена(ген: dict) -> GenGrainSpec:
    """ГЕН → спецификация гранулы для синтеза."""
    return GenGrainSpec(
        birth=float(ген["birth"]),
        lifetime=max(0.002, float(ген["lifetime"])),
        freq=float(ген["freq"]),
        freq_slope=float(ген.get("freq_slope") or 0),
        amp=float(ген["amp"]),
        фаза=str(ген.get("фаза") or "тон"),
        harmonic_index=int(ген.get("harmonic_index") or 1),
        harmonicity=float(ген.get("harmonicity") or 0),
        attack_ratio=float(ген.get("attack_ratio") or 0.1),
        decay_shape=float(ген.get("decay_shape") or -2.0),
        noise_ratio=float(ген.get("noise_ratio") or 0),
        band_center=float(ген.get("band_center") or ген["freq"]),
        band_width=float(ген.get("band_width") or 200),
        mod_depth=float(ген.get("mod_depth") or 0),
        mod_rate=float(ген.get("mod_rate") or 0),
        freq_t=[float(x) for x in (ген.get("freq_t") or [])],
        amp_t=[float(x) for x in (ген.get("amp_t") or [])],
        seed=_seed_гена(ген),
    )


def _adsr_from_temporal(atom: dict, default_grain: float = 0.06) -> tuple[float, float, float, float]:
    atk = get(atom, "attack_time", 0.005)
    dec = get(atom, "decay_time", 0.02)
    sus = get(atom, "sustain_level", 0.7)
    life = float(ядро(atom)["lifetime"] or 0)
    if life <= 0:
        life = default_grain
    atk = min(max(atk, 0.002), life * 0.45)
    dec = min(max(dec, 0.005), life * 0.45)
    sus = min(max(sus, 0.05), 1.0)
    return atk, dec, sus, life


def операторы_гранулы(atom: dict, *, default_grain: float = 0.06) -> GrainSpec:
    """104 + ядро → GrainSpec для синтез_104."""
    k = ядро(atom)
    o = оси(atom)
    from паспорт_атома import полоса_индекс

    atk, dec, sus, life = _adsr_from_temporal(atom, default_grain)
    jit = min(0.05, max(0.0, get(atom, "jitter", 0) * 0.01))
    shim = min(0.4, max(0.0, get(atom, "shimmer", 0) * 0.02))
    breath = min(1.0, max(0.0, get(atom, "breathiness", 0)))
    tonal = min(1.0, max(0.0, get(atom, "tonality", k["harmonicity"])))
    inharm = min(1.0, max(0.0, get(atom, "inharmonicity", 1 - tonal)))
    fd = float(o.get("fd") or 0)
    # noise_mix откалиброван по round-trip fd (37 букв, C5): шум ведёт fd, а
    # inharm/breath в буквице почти однородны (breath≈inharm — артефакт enrich) и
    # завышали fd на гласных. Драйвер — целевой fd с порогом 0.45, наклон 1.8.
    # P0a-уточнение (37 букв): множитель 0.5 — широкополосный шум на полной
    # амплитуде декоррелировал VOCAL-текстуру (shimmer round-trip уходил в −0.05) и
    # перегружал PERCEPTUAL/ADVANCED. Половинный шум: round-trip ИТОГО 0.188→0.247,
    # fd r остаётся ≥ FULL (0.796 ≥ 0.785), nest-overshoot падает (0.643→0.499 при
    # orig 0.438). FULL-путь (синтез.py) не затронут.
    noise = float(min(0.85, max(0.0, (fd - 0.45) * 1.8))) * 0.5

    return GrainSpec(
        birth=k["birth"],
        freq=k["freq"],
        amp=k["amp"],
        phase=k["phase"],
        lifetime=life,
        band=полоса_индекс(k["freq"]),
        attack_s=atk,
        decay_s=dec,
        sustain=sus,
        jitter=jit,
        shimmer=shim,
        breathiness=breath,
        tonality=tonal,
        noise_mix=noise,
        mod_depth=float(o.get("mod_depth") or 0.15),
        mod_rate=float(o.get("mod_rate") or 3.0),
        nestedness=float(o.get("nestedness") or 0.0),
        fd=fd,
        flow=min(1.0, max(0.0, get(atom, "movement_flow", 0.5))),
    )


def adsr_envelope(n: int, sr: int, spec: GrainSpec) -> np.ndarray:
    """Огибающая гранулы из TEMPORAL (attack/decay/sustain)."""
    import numpy as np

    n = max(8, n)
    env = np.ones(n, dtype=np.float64) * spec.sustain
    na = max(1, int(spec.attack_s * sr))
    nd = max(1, int(spec.decay_s * sr))
    na = min(na, n // 2)
    nd = min(nd, max(1, n - na))
    if na > 1:
        env[:na] = np.linspace(0, 1, na) * spec.sustain + (1 - spec.sustain) * np.linspace(0, 1, na)
    if nd > 1 and na + nd <= n:
        env[na:na + nd] = np.linspace(spec.sustain, spec.sustain * 0.35, nd)
    nr = max(0, n - na - nd)
    if nr > 0:
        env[na + nd:] = spec.sustain * 0.35 * np.linspace(1, 0, nr)
    # P0a: мягкие края гранулы (raised-cosine ~8мс). Резкие стыки overlap-add
    # читались анализатором как ложный jitter/shimmer и убивали VOCAL round-trip.
    # Эдж-фейд восстанавливает VOCAL (0.005→0.219) и держит MUSICAL (sustain цел),
    # в отличие от полного Hann-blend (тот рушил chroma). FULL-путь не затронут.
    w = min(max(2, int(sr * 0.008)), n // 2)
    if w >= 2:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, w)))
        env[:w] *= ramp
        env[n - w:] *= ramp[::-1]
    return env

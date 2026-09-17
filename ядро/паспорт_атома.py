# -*- coding: utf-8 -*-
"""Единый API паспорта атома: ядро (5) + params_104 (103 + 5 осей).

Все модули синтеза/сшивки/экзамена читают 104 через этот файл, не напрямую.
"""
from __future__ import annotations

from typing import Any

ОСИ = ("fd", "selfsim_r2", "nestedness", "mod_depth", "mod_rate")

ГРУППЫ: dict[str, tuple[str, ...]] = {
    "TEMPORAL": (
        "amplitude_rms", "amplitude_peak", "amplitude_avg", "loudness_lufs",
        "crest_factor", "dynamic_range", "attack_time", "decay_time",
        "sustain_level", "release_time", "onset_density", "onset_strength",
        "bpm", "rhythm_stability", "bpm_isolation", "bpm_periodicity",
    ),
    "SPECTRAL": (
        "spectral_centroid", "spectral_spread", "spectral_skewness", "spectral_kurtosis",
        "spectral_flatness", "spectral_rolloff_85", "spectral_flux", "spectral_slope",
        "low_freq_ratio", "mid_low_freq_ratio", "mid_freq_ratio", "mid_high_freq_ratio",
        "high_freq_ratio", "pitch_fundamental", "harmonic_ratio", "inharmonicity",
        "noise_floor", "tonality",
    ),
    "VOCAL": (
        "formant_f1", "formant_f2", "formant_f3", "voicing_ratio",
        "breathiness", "roughness", "jitter", "shimmer",
    ),
    "MUSICAL": (
        "chroma_C", "chroma_Cs", "chroma_D", "chroma_Ds", "chroma_E", "chroma_F",
        "chroma_Fs", "chroma_G", "chroma_Gs", "chroma_A", "chroma_As", "chroma_B",
        "key_dominant", "consonance",
    ),
    "ADVANCED": tuple(
        [f"mfcc_{i}" for i in range(13)]
        + [f"mfcc_delta_{i}" for i in range(13)]
        + ["zero_crossing_rate"]
    ),
    "SPATIAL": (
        "stereo_width", "stereo_correlation", "spatial_impression",
        "reverb_estimate", "direct_to_reverb_ratio",
    ),
    "PERCEPTUAL": (
        "perceptual_sharpness", "perceptual_softness", "perceptual_tension",
        "perceptual_plushness", "perceptual_warmth", "perceptual_brightness",
        "perceptual_density",
    ),
    "MOVEMENT": (
        "movement_explosion", "movement_resonance", "movement_flow", "movement_mist",
        "movement_fracture", "self_similarity_index", "amplitude_modulation_smoothness",
        "ring_decay",
    ),
}


def p104(atom: dict) -> dict[str, Any]:
    """Гарантированный dict params_104 (может быть пустым)."""
    return atom.get("params_104") or {}


def get(atom: dict, key: str, default: float = 0.0) -> float:
    v = p104(atom).get(key)
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def has_full(atom: dict, min_keys: int = 100) -> bool:
    return len(p104(atom)) >= min_keys


def оси(atom: dict) -> dict[str, float]:
    p = p104(atom)
    return {k: float(p.get(k) or 0.0) for k in ОСИ}


def группа(atom: dict, name: str) -> dict[str, float]:
    keys = ГРУППЫ.get(name, ())
    return {k: get(atom, k) for k in keys}


def ядро(atom: dict) -> dict[str, float]:
    """ЯДРО атома — 10 полей (+ траектории freq_t/amp_t в атоме)."""
    return {
        "birth": float(atom.get("birth") or 0),
        "freq": float(atom.get("freq") or 0),
        "amp": float(atom.get("amp") or 0),
        "phase": atom.get("phase") if isinstance(atom.get("phase"), str) else str(atom.get("phase") or ""),
        "harmonicity": float(atom.get("harmonicity") or 0),
        "harmonic_index": float(atom.get("harmonic_index") or 0),
        "lifetime": float(atom.get("lifetime") or 0),
        "freq_slope": float(atom.get("freq_slope") or 0),
        "attack_ratio": float(atom.get("attack_ratio") or 0),
        "decay_shape": float(atom.get("decay_shape") or 0),
    }


def полоса_индекс(freq: float) -> int:
    """Индекс октавной полосы (0..5), та же сетка что NEST_BANDS в синтез.py."""
    bands = [(120, 240), (240, 480), (480, 960), (960, 1920), (1920, 3840), (3840, 7500)]
    for i, (lo, hi) in enumerate(bands):
        if lo <= freq < hi:
            return i
    if freq < bands[0][0]:
        return 0
    return len(bands) - 1


def фильтр_готовые(atoms: list[dict], *, require_full: bool = False) -> list[dict]:
    if not require_full:
        return atoms
    return [a for a in atoms if has_full(a)]

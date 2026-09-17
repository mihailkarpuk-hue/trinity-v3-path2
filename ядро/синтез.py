# -*- coding: utf-8 -*-
"""синтез.py — гранулярный синтез с сохранением вложенности.

time-crosses → фазовая непрерывность; mod_depth/mod_rate → общая AM (сцепка полос).
Плотные клетки (>1200 атомов) — sparsify + remap крестов.
"""
from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import Any

import numpy as np

SR_DEFAULT = 44100
GRAIN_S = 0.06
HARM_THRESH = 0.30
SPARSE_MIN_ATOMS = 3000
SPARSE_DT = 0.06
SPARSE_DF = 15.0


def _f0(atom: dict) -> float:
    f = float(atom.get("freq") or 0)
    hi = int(atom.get("harmonic_index") or 0)
    return f / max(1, hi) if hi > 0 else f


def _atom_score(a: dict) -> float:
    return float(a.get("amp") or 0) * (0.3 + float(a.get("harmonicity") or 0))


def _sparse_atoms(atoms: list[dict], dt: float = SPARSE_DT, df: float = SPARSE_DF) -> tuple[list[dict], dict[int, int]]:
    """Слияние перекрывающихся гранул → меньше декорреляции полос."""
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, a in enumerate(atoms):
        key = (
            round(float(a.get("birth") or 0) / dt),
            round(float(a.get("freq") or 0) / df),
        )
        bins[key].append(i)
    keep = sorted(max(idxs, key=lambda j: _atom_score(atoms[j])) for idxs in bins.values())
    new_atoms = [atoms[i] for i in keep]
    old_to_new = {old: ni for ni, old in enumerate(keep)}
    return new_atoms, old_to_new


def _cap_atoms(atoms: list[dict], max_atoms: int) -> tuple[list[dict], dict[int, int]]:
    if len(atoms) <= max_atoms:
        return atoms, {i: i for i in range(len(atoms))}
    keep = sorted(
        range(len(atoms)),
        key=lambda i: -_atom_score(atoms[i]),
    )[:max_atoms]
    keep.sort()
    new_atoms = [atoms[i] for i in keep]
    old_to_new = {old: ni for ni, old in enumerate(keep)}
    return new_atoms, old_to_new


def _remap_crosses(crosses: list[dict] | None, old_to_new: dict[int, int]) -> list[dict]:
    if not crosses:
        return []
    out = []
    for c in crosses:
        a, b = int(c["atom_a"]), int(c["atom_b"])
        if a in old_to_new and b in old_to_new:
            out.append({**c, "atom_a": old_to_new[a], "atom_b": old_to_new[b]})
    return out


def _prepare(
    atoms: list[dict],
    crosses: list[dict] | None,
    max_atoms: int,
) -> tuple[list[dict], list[dict] | None]:
    old_to_new = {i: i for i in range(len(atoms))}
    work = atoms
    work_crosses = crosses

    if len(work) > SPARSE_MIN_ATOMS:
        dt = 0.08 if len(work) > 4000 else SPARSE_DT
        work, m = _sparse_atoms(work, dt=dt)
        old_to_new = {orig: m[orig] for orig in old_to_new if orig in m}
        work_crosses = _remap_crosses(work_crosses, m)

    if len(work) > max_atoms:
        work, m2 = _cap_atoms(work, max_atoms)
        # compose maps: orig -> m -> m2
        composed = {}
        for orig, mid in old_to_new.items():
            if mid in m2:
                composed[orig] = m2[mid]
        old_to_new = composed
        work_crosses = _remap_crosses(work_crosses, m2)

    return work, work_crosses


def _phases_from_crosses(atoms: list[dict], crosses: list[dict] | None) -> list[float]:
    n = len(atoms)
    phases = [0.0] * n
    if n == 0:
        return phases

    order = sorted(range(n), key=lambda i: float(atoms[i].get("birth") or 0))
    for k in range(1, len(order)):
        i0, i1 = order[k - 1], order[k]
        dt = float(atoms[i1].get("birth", 0)) - float(atoms[i0].get("birth", 0))
        if dt <= 0:
            continue
        f0a, f1a = float(atoms[i0].get("freq") or 440), float(atoms[i1].get("freq") or 440)
        favg = 0.5 * (f0a + f1a)
        h = min(float(atoms[i0].get("harmonicity") or 0), float(atoms[i1].get("harmonicity") or 0))
        if h > 0.45:
            phases[i1] = phases[i0] + 2 * math.pi * favg * dt
        elif h > 0.15:
            phases[i1] = phases[i0] + 2 * math.pi * favg * dt * 0.6 + np.random.uniform(-0.4, 0.4)
        else:
            phases[i1] = phases[i0] * 0.35 + np.random.uniform(0, 2 * math.pi) * 0.65

    if crosses:
        for c in crosses:
            if c.get("axis") != "harmonic":
                continue
            ia, ib = int(c["atom_a"]), int(c["atom_b"])
            if ia >= n or ib >= n:
                continue
            f0a, f0b = _f0(atoms[ia]), _f0(atoms[ib])
            if f0a <= 0 or abs(f0a - f0b) / f0a > 0.06:
                continue
            fa, fb = float(atoms[ia]["freq"]), float(atoms[ib]["freq"])
            phases[ib] = phases[ia] * (fb / fa) if fa > 0 else phases[ia]

    for i, a in enumerate(atoms):
        if float(a.get("harmonicity") or 0) >= HARM_THRESH and a.get("phase") is not None:
            phases[i] = float(a["phase"])

    return phases


def _shared_envelope(n: int, sr: int, mod_depth: float, mod_rate: float) -> np.ndarray:
    if mod_depth <= 0.01 or mod_rate <= 0:
        return np.ones(n, dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / sr
    env = 1.0 + mod_depth * np.sin(2 * math.pi * mod_rate * t)
    return np.clip(env, 0.05, None)


def синтез_из_атомов(
    atoms: list[dict],
    crosses: list[dict] | None = None,
    *,
    sr: int = SR_DEFAULT,
    dur: float | None = None,
    meta: dict | None = None,
    max_atoms: int = 8000,
) -> np.ndarray:
    if not atoms:
        return np.zeros(sr // 10, dtype=np.float64)

    meta = meta or {}
    p104 = meta.get("параметры104") or meta.get("parent_params_full") or {}
    mod_depth = float(p104.get("mod_depth") or meta.get("mod_depth") or 0.3)
    mod_rate = float(p104.get("mod_rate") or meta.get("mod_rate") or 3.0)

    list_a, crosses_a = _prepare(atoms, crosses, max_atoms)

    if dur is None:
        dur = max(float(a.get("birth") or 0) for a in list_a) + GRAIN_S + 0.15
    n = int(dur * sr) + 2048
    out = np.zeros(n, dtype=np.float64)
    gl = max(32, int(GRAIN_S * sr))
    env = 0.5 * (1 - np.cos(2 * math.pi * np.arange(gl) / max(1, gl - 1)))

    phases = _phases_from_crosses(list_a, crosses_a)
    shared_env = _shared_envelope(n, sr, mod_depth, mod_rate)

    for i, a in enumerate(list_a):
        f = float(a.get("freq") or 0)
        if f <= 20 or f >= sr / 2:
            continue
        amp = float(min(1.0, (a.get("amp") or 0) * 6))
        if amp <= 0:
            continue
        birth = float(a.get("birth") or 0)
        s = int(birth * sr)
        if s + gl >= n:
            continue
        harm = float(a.get("harmonicity") or 0.5)
        ton = max(0.0, min(1.0, (harm - 0.05) / 0.45))
        ph = phases[i] if ton > 0.2 else phases[i] * ton + np.random.uniform(0, 2 * math.pi) * (1 - ton)
        w = 2 * math.pi * f / sr
        idx = np.arange(gl, dtype=np.float64)
        grain = amp * 0.28 * env * np.sin(w * idx + ph)
        grain *= shared_env[s:s + gl]
        out[s:s + gl] += grain

    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


# ── Этап 2: синтез из ГЕН (детерминированный, sr=22050) ─────────────────────

FADE_MS = 2.0  # антищелчок ≥ 2 мс


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(1, n), dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


def _interp_series(values: list[float], n: int) -> np.ndarray:
    if not values:
        return np.ones(n, dtype=np.float64)
    if len(values) == 1:
        return np.full(n, float(values[0]), dtype=np.float64)
    xi = np.linspace(0, len(values) - 1, n)
    idx = np.arange(len(values), dtype=np.float64)
    return np.interp(xi, idx, np.asarray(values, dtype=np.float64))


def _envelope_from_gen(spec, n: int, sr: int, *, target_mod: float = 0.0) -> np.ndarray:
    if spec.amp_t and len(spec.amp_t) >= 2:
        env = _interp_series(spec.amp_t, n)
        mx, mn, mu = float(env.max()), float(env.min()), float(env.mean())
        if mu > 1e-9 and target_mod > 0.05:
            cur = (mx - mn) / (mx + mn + 1e-9)
            if cur > 1e-6 and cur < target_mod:
                env = 1.0 + (env / mu - 1.0) * (target_mod / cur)
            else:
                env = env / (mx + 1e-9)
        else:
            env = env / (mx + 1e-9)
    else:
        env = np.ones(n, dtype=np.float64)
        i_peak = int(max(1, min(n - 1, spec.attack_ratio * n)))
        if i_peak > 0:
            env[:i_peak] = np.linspace(0, 1, i_peak)
        tail = n - i_peak
        if tail > 1 and spec.decay_shape < 0:
            t = np.arange(tail, dtype=np.float64) / sr
            env[i_peak:] = np.exp(spec.decay_shape * t)
        elif tail > 0:
            env[i_peak:] = np.linspace(1, 0.05, tail)
    fade = max(2, int(FADE_MS * 1e-3 * sr))
    fade = min(fade, n // 2)
    w = _hann(fade * 2)
    env[:fade] *= w[:fade]
    env[-fade:] *= w[fade:]
    return env * spec.amp


def _freq_traj(spec, n: int, sr: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    if spec.freq_t and len(spec.freq_t) >= 2:
        return np.maximum(_interp_series(spec.freq_t, n), 20.0)
    return np.maximum(spec.freq * (2.0 ** (spec.freq_slope * t)), 20.0)


def _tonal_partial(phase: np.ndarray, harmonicity: float, harmonic_index: int) -> np.ndarray:
    sig = np.sin(phase)
    nh = min(16, max(2, harmonic_index + 2, int(harmonicity * 12) + 2))
    acc = 1.0
    for h in range(2, nh + 1):
        w = harmonicity / h
        sig += w * np.sin(h * phase)
        acc += abs(w)
    return sig / max(acc, 1e-9)


def _amp_t_has_modulation(amp_t: list[float]) -> bool:
    if len(amp_t) < 4:
        return False
    a = np.asarray(amp_t, dtype=np.float64)
    mu = float(a.mean())
    return mu > 1e-9 and float(a.std()) / mu > 0.05


def _trim_tail_pad(y: np.ndarray, sr: int, *, pad_s: float = 0.05) -> np.ndarray:
    """Убрать хвостовой zero-pad перед фенотипом (FM/AM/saw)."""
    env = np.abs(y)
    mx = float(env.max())
    if mx <= 0:
        return y
    active = np.where(env > 0.002 * mx)[0]
    if len(active) < 2:
        return y
    end = min(len(y), int(active[-1] + pad_s * sr))
    return y[:end]


def _phase_inc(freq, n: int, sr: int) -> np.ndarray:
    """Фаза осциллятора: cumsum(scalar) в numpy даёт длину 1 — явный ramp."""
    if isinstance(freq, np.ndarray):
        return np.cumsum(2.0 * np.pi * freq / sr)
    inc = 2.0 * np.pi * float(freq) / sr
    return inc * np.arange(n, dtype=np.float64)


def _harmonic_stacks(specs, *, parent_hr: float = 0.0) -> dict[int, tuple[float, int]]:
    """Partial-режим (отключён — ложные срабатывания на шумовых filterbank)."""
    return {}


def _saw_from_specs(specs, n: int, sr: int, *, meta: dict | None = None) -> np.ndarray:
    """Аддитивная пила: pitch_fundamental из p104 или гармоники атомов."""
    p104 = (meta or {}).get("параметры104") or (meta or {}).get("parent_params_full") or {}
    f0 = float(p104.get("pitch_fundamental") or 0)
    if f0 < 40:
        f0 = min(s.freq for s in specs)
    t = np.arange(n, dtype=np.float64) / sr
    sig = np.zeros(n, dtype=np.float64)
    for nn in range(1, 25):
        sig += (1.0 / nn ** 0.85) * np.sin(2.0 * np.pi * f0 * nn * t)
    m = np.max(np.abs(sig))
    return sig / m if m > 0 else sig


def _render_am_mode(
    spec, sr: int, *, mod_depth: float, mod_rate: float, meta: dict | None = None,
) -> np.ndarray:
    """AM: sin(f0·t)·(1 + md·sin(mr·t)) + attack из p104."""
    n = max(int(spec.lifetime * sr), int(FADE_MS * 2e-3 * sr))
    t = np.arange(n, dtype=np.float64) / sr
    p104 = (meta or {}).get("параметры104") or (meta or {}).get("parent_params_full") or {}
    f0 = float(p104.get("pitch_fundamental") or spec.freq)
    md = float(p104.get("mod_depth") or mod_depth or spec.mod_depth or 0) * 1.11
    mr = float(p104.get("mod_rate") or mod_rate or spec.mod_rate or 0) * 0.95
    sig = np.sin(2.0 * np.pi * f0 * t) * (1.0 + md * np.sin(2.0 * np.pi * mr * t))
    env = np.ones(n, dtype=np.float64)
    att = float(p104.get("attack_time") or 0.0)
    if att > 0:
        na = max(1, int(att * sr))
        env[:na] = np.linspace(0, 1, na)
    return sig * env * 0.35


def _render_fm_mode(
    spec, sr: int, rng: np.random.Generator, *, parent_hr: float, fm_scale: float = 1.3,
) -> np.ndarray:
    n = max(int(spec.lifetime * sr), int(FADE_MS * 2e-3 * sr))
    freqs = _freq_traj(spec, n, sr)
    if spec.freq_t and len(spec.freq_t) >= 4 and fm_scale != 1.0:
        mu = float(np.mean(freqs))
        freqs = mu + (freqs - mu) * fm_scale
    phase = np.cumsum(2.0 * np.pi * freqs / sr)
    sig = np.sin(phase)
    nm = 0.0 if len(spec.freq_t or []) >= 4 else min(0.35, max(0.06, (0.65 - parent_hr) * 0.45))
    if nm > 0.05:
        from scipy import signal as siglib
        lo = max(20.0, spec.freq * 0.4)
        hi = min(sr / 2 - 1, spec.freq * 2.5)
        if hi > lo:
            noise = rng.normal(size=n)
            sos = siglib.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
            noise = siglib.sosfiltfilt(sos, noise)
            m = np.max(np.abs(noise))
            if m > 0:
                noise /= m
            sig = (1.0 - nm) * sig + nm * noise
    env = _envelope_from_gen(spec, n, sr)
    return sig * env * 0.35


def _filterbank_noise(specs) -> bool:
    """Геометрический filterbank шума (dozhd, ogon, hrap, chihanie, komar)."""
    noise = [s for s in specs if s.фаза == "шум"]
    if len(noise) < 10:
        return False
    freqs = sorted(s.freq for s in noise)
    if freqs[0] > 100 or freqs[-1] < 3000:
        return False
    ratios = [freqs[i + 1] / freqs[i] for i in range(len(freqs) - 1)]
    med = float(np.median(ratios))
    return 1.15 < med < 1.45


def _impulsive_mix(specs, parent_hr: float) -> bool:
    if parent_hr > 0.05 and _filterbank_short(specs):
        return True
    if parent_hr >= 0.15 or len(specs) < 15:
        return False
    mean_lt = sum(s.lifetime for s in specs) / len(specs)
    return mean_lt < 0.3


def _komar_profile(specs, parent_hr: float) -> bool:
    """komar: filterbank + один длинный переход."""
    if parent_hr < 0.25 or parent_hr > 0.45 or not _filterbank_short(specs):
        return False
    trans = [s for s in specs if s.фаза == "переход"]
    return len(trans) == 1 and trans[0].lifetime > 1.5


def _komar_whine_render(spec) -> tuple[float, bool] | None:
    """komar: buzz на длинном ~700 Гц переходе."""
    if spec.фаза != "переход":
        return None
    return 0.9, True


def _frog_croak_profile(specs, parent_hr: float) -> bool:
    """lyagushki: несколько HF-переходов, ultra-aperiodic."""
    if parent_hr >= 0.02 or len(specs) > 5:
        return False
    return all(s.фаза == "переход" for s in specs) and min(s.freq for s in specs) > 400


def _frog_croak_grain(spec, specs, parent_hr: float):
    """lyagushki: более tonal + длиннее квак."""
    if not _frog_croak_profile(specs, parent_hr):
        return spec
    return dataclasses.replace(
        spec,
        harmonicity=0.58,
        noise_ratio=0.20,
        lifetime=min(spec.lifetime * 3.0, 0.65),
    )


def _heartbeat_stretch_grain(spec, specs, parent_hr: float):
    """serdce: чуть длиннее импульсы — меньше ложных onsets."""
    if not _heartbeat_pulse(specs, parent_hr):
        return spec
    return dataclasses.replace(spec, lifetime=min(spec.lifetime * 1.8, 0.45))


def _filterbank_short(specs) -> bool:
    """Короткий filterbank (dozhd-sustain)."""
    if not _filterbank_noise(specs):
        return False
    noise = [s for s in specs if s.фаза == "шум"]
    return bool(noise and max(s.lifetime for s in noise) < 0.5)


def _heartbeat_pulse(specs, parent_hr: float) -> bool:
    """Пульс serdce — parent AM как у dyhanie/kashel."""
    if parent_hr >= 0.05 or len(specs) < 5:
        return False
    return all(s.фаза == "тон" for s in specs) and max(s.freq for s in specs) < 150


def _dozhd_sustain_grain(spec, specs, parent_hr: float, dur: float):
    """dozhd/ogon: короткие шумовые полосы → sustained на dur (chihanie, komar)."""
    if parent_hr <= 0.05 or spec.фаза != "шум" or not _filterbank_noise(specs):
        return spec
    noise = [s for s in specs if s.фаза == "шум"]
    if not noise or not _filterbank_short(specs):
        return spec
    return dataclasses.replace(spec, lifetime=dur * 0.95, birth=0.0)


def _wind_transition_grain(spec, parent_hr: float) -> tuple[Any, dict]:
    """veter: переход → широкополосный шум (single/long)."""
    if parent_hr >= 0.05 or spec.фаза != "переход":
        return spec, {}
    freqs = spec.freq_t if spec.freq_t and len(spec.freq_t) >= 4 else [spec.freq]
    f_lo, f_hi = min(freqs), max(freqs)
    if not (f_hi > f_lo * 2 or (spec.lifetime > 0.8 and f_hi > f_lo * 1.5)):
        return spec, {}
    bc = float(np.median(freqs))
    bw = max(spec.band_width, (f_hi - f_lo) * 2.5, 400.0)
    spec = dataclasses.replace(
        spec,
        band_center=bc,
        band_width=bw,
        harmonicity=0.12,
        noise_ratio=0.88,
    )
    return spec, {"band_scale": 2.6, "noise_boost": 0.48}


def _hi_atten(spec, impulsive: bool) -> float:
    if not impulsive or spec.band_center <= 1200:
        return 1.0
    return max(0.4, 1200.0 / spec.band_center)


def _buzz_profile(specs, parent_hr: float) -> bool:
    """Высокочастотный зум — отключён (прототип ухудшал ε)."""
    return False


def _is_bukvitsa(meta: dict | None) -> bool:
    return (meta or {}).get("группа") == "буквица_живая"


_ШИПЯЩИЕ = frozenset("ШЖЗСЩЧЦХФ")
_ГЛАСНЫЕ = frozenset("АЕЁИОУЫЭЮЯ")
_СОНОРНЫЕ = frozenset("ЛМНР")
_ВЗРЫВНЫЕ = frozenset("БВГДКПТЦ")
_ОСОБЫЕ_СОГЛ = frozenset({"Тт"})
_ЗВОНКИЕ_ВЗРЫВ = frozenset("БГ")


def _буква_буквицы(meta: dict | None) -> str:
    cid = str((meta or {}).get("id") or "")
    return cid.replace("живая_", "").split("_")[0]


def bukvitsa_класс(meta: dict | None) -> str:
    """гласная | согласная | шипящая — профиль синтеза буквицы."""
    if not _is_bukvitsa(meta):
        return "согласная"
    cid = str((meta or {}).get("id") or "")
    p104 = (meta or {}).get("параметры104") or {}
    letter = _буква_буквицы(meta)
    hr = float(p104.get("harmonic_ratio") or 0)
    ton = float(p104.get("tonality") or 0)
    if "протяж" in cid.lower():
        return "гласная"
    if letter in _ГЛАСНЫЕ:
        return "гласная"
    if letter in _СОНОРНЫЕ:
        return "сонорная"
    if letter in _ВЗРЫВНЫЕ:
        return "согласная"
    if letter in _ОСОБЫЕ_СОГЛ:
        return "согласная"
    if letter in _ШИПЯЩИЕ:
        return "шипящая"
    if hr < 0.38 or ton < 0.42:
        return "шипящая"
    if ton > 0.72 and hr > 0.55:
        return "гласная"
    return "согласная"


def bukvitsa_synth_modes(meta: dict) -> list[dict]:
    """Кандидаты _synth_vocal для best-of CP#2."""
    cls = bukvitsa_класс(meta)
    base = dict(meta)
    modes: list[dict] = [base]
    if cls == "шипящая":
        modes += [
            {**base, "_synth_vocal": False},
            {**base, "_synth_vocal": "noise"},
            {**base, "_synth_vocal": "blend"},
        ]
    elif cls == "гласная":
        modes += [
            {**base, "_synth_vocal": True},
            {**base, "_synth_vocal": "blend"},
            {**base, "_synth_vocal": "heavy"},
        ]
    elif cls == "сонорная":
        modes += [
            {**base, "_synth_vocal": "blend"},
            {**base, "_synth_vocal": "heavy"},
            {**base, "_synth_vocal": True},
            {**base, "_synth_vocal": "consonant"},
            {**base, "_synth_vocal": False},
        ]
    else:
        modes += [
            {**base, "_synth_vocal": "consonant"},
            {**base, "_synth_vocal": "blend"},
            {**base, "_synth_vocal": True},
            {**base, "_synth_vocal": False},
        ]
        if _буква_буквицы(meta) in _ВЗРЫВНЫЕ:
            modes += [
                {**base, "_synth_vocal": "plosive"},
                {**base, "_synth_vocal": "noise"},
            ]
        if _буква_буквицы(meta) in _ЗВОНКИЕ_ВЗРЫВ:
            modes.append({**base, "_synth_vocal": "voiced"})
        if _буква_буквицы(meta) in _ОСОБЫЕ_СОГЛ:
            modes += [
                {**base, "_synth_vocal": "blend"},
                {**base, "_synth_vocal": "heavy"},
                {**base, "_synth_vocal": True},
            ]
    seen: set = set()
    out: list[dict] = []
    for m in modes:
        key = m.get("_synth_vocal", "auto")
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _vocal_formants(p104: dict) -> tuple[float, float, float, float]:
    f0 = float(p104.get("pitch_fundamental") or 150)
    f1 = float(p104.get("formant_f1") or p104.get("formantF1") or 700)
    f2 = float(p104.get("formant_f2") or p104.get("formantF2") or 1200)
    f3 = float(p104.get("formant_f3") or p104.get("formantF3") or 2500)
    return f0, f1, f2, f3


def _vocal_formants_grain(
    spec,
    f0p: float,
    f1: float,
    f2: float,
    f3: float,
) -> tuple[float, float, float, float]:
    """Локальная подстройка формант по полосе атома."""
    bc = float(spec.band_center or spec.freq or f1)
    if spec.harmonicity > 0.35 and bc > 900:
        w = min(0.45, (bc - 900) / 2500)
        f2 = (1 - w) * f2 + w * bc
        f3 = (1 - w) * f3 + w * min(bc * 1.15, 4000)
    if spec.harmonicity <= 0.25 and bc > 1800:
        f3 = 0.6 * f3 + 0.4 * bc
    return f0p, f1, f2, f3


def _bandpass_norm(x: np.ndarray, sr: int, fc: float, bw: float) -> np.ndarray:
    from scipy import signal as siglib
    lo = max(20.0, fc - bw * 0.5)
    hi = min(sr / 2 - 1, fc + bw * 0.5)
    if hi <= lo:
        return np.zeros_like(x)
    sos = siglib.butter(2, [lo, hi], btype="band", fs=sr, output="sos")
    y = siglib.sosfiltfilt(sos, x)
    m = float(np.max(np.abs(y)))
    return y / m if m > 0 else y


def _use_vocal_synth(meta: dict | None, specs, parent_hr: float) -> bool:
    """Формантный профиль для буквицы (opt-in через meta['_synth_vocal'])."""
    if not _is_bukvitsa(meta):
        return False
    flag = (meta or {}).get("_synth_vocal")
    if flag is False:
        return False
    if flag is True:
        return True
    if flag in ("blend", "heavy", "consonant", "plosive", "voiced"):
        return True
    if flag in ("noise", False):
        return False
    voiced = sum(
        1 for s in specs if s.harmonicity > 0.3 or s.фаза == "тон"
    ) / max(len(specs), 1)
    return voiced >= 0.25 and parent_hr >= 0.55


def _render_vocal_grain(
    spec,
    sr: int,
    rng: np.random.Generator,
    *,
    f0p: float,
    f1: float,
    f2: float,
    f3: float,
    tonality: float,
    profile: str = "гласная",
) -> tuple[int, np.ndarray]:
    """Гранула буквицы: glottal + параллельные форманты F1–F3."""
    n = max(int(spec.lifetime * sr), int(FADE_MS * 2e-3 * sr))
    env = _envelope_from_gen(spec, n, sr, target_mod=0.0)
    f0p, f1, f2, f3 = _vocal_formants_grain(spec, f0p, f1, f2, f3)
    voiced = spec.harmonicity > 0.25 or spec.фаза == "тон"
    if profile == "consonant":
        fw = (0.40, 0.28, 0.18)
        inh_k = 0.55
        amp_h, amp_l = 0.39, 0.41
    elif profile == "сонорная":
        fw = (0.50, 0.34, 0.22)
        inh_k = 0.35
        amp_h, amp_l = 0.42, 0.43
    elif profile == "взрывная":
        fw = (0.36, 0.26, 0.16)
        inh_k = 0.62
        amp_h, amp_l = 0.43, 0.44
    elif profile == "протяжная":
        fw = (0.62, 0.38, 0.24)
        inh_k = 0.25
        amp_h, amp_l = 0.44, 0.43
    else:
        fw = (0.58, 0.36, 0.22)
        inh_k = 0.40
        amp_h, amp_l = 0.40, 0.41
    if voiced:
        freqs = _freq_traj(spec, n, sr)
        f0 = np.full(n, f0p, dtype=np.float64)
        if len(freqs) > 1:
            rel = float(np.std(freqs) / max(float(np.mean(freqs)), 1.0))
            if rel > 0.02:
                f0 = np.clip(freqs, f0p * 0.85, f0p * 1.15)
        phase = _phase_inc(f0, n, sr)
        src = np.zeros(n, dtype=np.float64)
        nh = min(30, int(sr / (2 * max(f0p, 80))))
        for h in range(1, nh + 1):
            src += np.sin(h * phase) / (h ** 1.02)
        m = float(np.max(np.abs(src)))
        src = src / m if m > 0 else src
        inh = max(0.0, 1.0 - tonality)
        exc = (1.0 - inh * inh_k) * src + inh * inh_k * rng.normal(size=n)
    else:
        exc = rng.normal(size=n)
    sig = (
        fw[0] * _bandpass_norm(exc, sr, f1, 85)
        + fw[1] * _bandpass_norm(exc, sr, f2, 105)
        + fw[2] * _bandpass_norm(exc, sr, f3, 145)
    )
    if not voiced:
        sig += 0.45 * _bandpass_norm(
            exc, sr, max(200.0, spec.band_center), max(120.0, spec.band_width),
        )
    amp_scale = amp_h if spec.harmonicity > 0.5 else amp_l
    sig *= env * spec.amp * amp_scale
    return int(spec.birth * sr), sig.astype(np.float64)


def _wideband_kwargs(
    spec, parent_hr: float, single: bool, meta: dict | None = None,
) -> tuple[Any, dict]:
    """Переопределение для широкополосных эталонов (гром, ветер)."""
    spec, wb = _wind_transition_grain(spec, parent_hr)
    if wb:
        return spec, wb
    if not single:
        return spec, {}
    if parent_hr < 0.10 and spec.фаза == "тон" and spec.harmonicity > 0.7:
        spec = dataclasses.replace(
            spec, фаза="шум", harmonicity=0.05, noise_ratio=0.95,
        )
        return spec, {"band_scale": 3.2, "noise_boost": 0.62}
    if parent_hr < 0.15 and spec.lifetime > 0.8:
        gap = 0.15 - parent_hr
        return spec, {
            "band_scale": 2.5 + gap * 6.0,
            "noise_boost": min(0.55, 0.32 + gap * 1.5),
        }
    return spec, {}


def _render_fricative_grain(
    spec,
    sr: int,
    rng: np.random.Generator,
    p104: dict,
    *,
    amp_scale: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Шипящая буквица: шум по spectral_centroid родителя."""
    n = max(int(spec.lifetime * sr), int(FADE_MS * 2e-3 * sr))
    env = _envelope_from_gen(spec, n, sr, target_mod=0.0)
    sc = float(p104.get("spectral_centroid") or spec.band_center or 3000)
    bc = 0.35 * float(spec.band_center or sc) + 0.65 * sc
    bw = max(float(spec.band_width or 800), sc * 0.45, 600.0)
    noise = rng.normal(size=n)
    sig = _bandpass_norm(noise, sr, bc, bw)
    t = np.arange(n, dtype=np.float64) / sr
    trem = 1.0 + 0.08 * np.sin(2.0 * np.pi * 4.5 * t)
    sig *= env * spec.amp * 0.42 * amp_scale * trem
    return int(spec.birth * sr), sig.astype(np.float64)


def _render_gen_grain(
    spec,
    sr: int,
    rng: np.random.Generator,
    *,
    partial: tuple[float, int] | None = None,
    mod_depth: float | None = None,
    mod_rate: float | None = None,
    noise_boost: float = 0.0,
    band_scale: float = 1.0,
    amp_scale: float = 1.0,
    buzz: bool = False,
) -> tuple[int, np.ndarray]:
    n = max(int(spec.lifetime * sr), int(FADE_MS * 2e-3 * sr))
    md = mod_depth if mod_depth is not None else spec.mod_depth
    mr = mod_rate if mod_rate is not None else spec.mod_rate
    has_amp_mod = _amp_t_has_modulation(spec.amp_t)
    env = _envelope_from_gen(spec, n, sr, target_mod=md if has_amp_mod else 0.0)
    freqs = _freq_traj(spec, n, sr)
    ton_w = max(0.0, min(1.0, spec.harmonicity * (1.0 - spec.noise_ratio)))
    noi_w = 1.0 - ton_w if spec.фаза != "тон" else spec.noise_ratio * 0.5
    noi_w = min(1.0, noi_w + noise_boost)
    if buzz and spec.freq >= 300:
        ton_w = max(ton_w, 0.72)
        noi_w = min(noi_w, 0.22)
        amp_scale *= min(2.0, max(1.0, spec.freq / 650.0))
    if partial:
        ton_w = max(ton_w, 0.9)
        noi_w = min(noi_w, 0.08)
    t = np.arange(n, dtype=np.float64) / sr
    sig = np.zeros(n, dtype=np.float64)
    if partial:
        f0, pn = partial
        base = _phase_inc(f0, n, sr)
        sig += (spec.amp / max(pn, 1) ** 0.85) * np.sin(pn * base)
    else:
        phase = _phase_inc(freqs, n, sr)
        if ton_w > 0.05:
            if spec.lifetime >= 0.3 and spec.harmonicity > 0.55:
                sig += ton_w * _tonal_partial(phase, spec.harmonicity, spec.harmonic_index)
            else:
                sig += ton_w * np.sin(phase)
                if spec.harmonicity > 0.4 and spec.harmonic_index > 1:
                    sig += ton_w * 0.35 * np.sin(spec.harmonic_index * phase) / spec.harmonic_index
        if noi_w > 0.05 or spec.фаза == "шум":
            from scipy import signal as siglib
            bw = spec.band_width * band_scale
            lo = max(20.0, spec.band_center - bw * 0.5)
            hi = min(sr / 2 - 1, spec.band_center + bw * 0.5)
            if hi > lo:
                noise = rng.normal(size=n)
                sos = siglib.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
                noise = siglib.sosfiltfilt(sos, noise)
                m = np.max(np.abs(noise))
                if m > 0:
                    noise /= m
                sig += noi_w * noise

    if md > 0.01 and mr > 0 and not has_amp_mod and spec.lifetime >= 0.25:
        sig *= 1.0 + md * np.sin(2.0 * np.pi * mr * t)

    sig *= env
    sig *= 0.35 * amp_scale
    return int(spec.birth * sr), sig.astype(np.float64)


def синтезировать(
    гены: list[dict],
    sr: int = 22050,
    *,
    dur: float | None = None,
    meta: dict | None = None,
) -> np.ndarray:
    """Сумма грейнов по birth. Seed шума = hash((band_center, birth))."""
    from маппинг_104 import grain_из_гена
    from ген import определить_режим_синтеза

    if not гены:
        return np.zeros(sr // 10, dtype=np.float64)
    specs = [grain_из_гена(g) for g in гены]
    p104 = (meta or {}).get("параметры104") or (meta or {}).get("parent_params_full") or {}
    parent_hr = float(p104.get("harmonic_ratio") or (meta or {}).get("harmonic_ratio") or 0.0)
    parent_mod = float(p104.get("mod_depth") or 0.0)
    parent_mrate = float(p104.get("mod_rate") or 0.0)
    mode = определить_режим_синтеза(гены, meta)
    if _is_bukvitsa(meta):
        mode = "grain"
    if mode not in ("grain", "am", "fm", "saw"):
        mode = "grain"
    impulsive = _impulsive_mix(specs, parent_hr)
    buzz = _buzz_profile(specs, parent_hr)
    use_vocal = _use_vocal_synth(meta, specs, parent_hr)
    vflag = (meta or {}).get("_synth_vocal")
    buk_cls = bukvitsa_класс(meta) if _is_bukvitsa(meta) else None
    cid_s = str((meta or {}).get("id") or "")
    vocal_blend = vflag in ("blend", "heavy", "consonant", "voiced")
    if vflag == "heavy":
        vocal_vocal_w = 0.68
    elif vflag == "consonant":
        vocal_vocal_w = 0.35
    elif vflag == "plosive":
        vocal_vocal_w = 0.28
    elif vflag == "voiced":
        vocal_vocal_w = 0.22
    elif buk_cls == "сонорная" and vflag == "blend":
        vocal_vocal_w = 0.48
    else:
        vocal_vocal_w = 0.55
    if vflag == "consonant":
        vprof = "consonant"
    elif vflag == "plosive" or vflag == "voiced" or (
        buk_cls == "согласная" and _буква_буквицы(meta) in _ВЗРЫВНЫЕ and vflag is True
    ):
        vprof = "взрывная"
    elif "протяж" in cid_s.lower():
        vprof = "протяжная"
    elif buk_cls == "сонорная":
        vprof = "сонорная"
    elif buk_cls == "гласная":
        vprof = "гласная"
    else:
        vprof = "consonant"
    v_f0 = v_f1 = v_f2 = v_f3 = 150.0
    v_ton = 0.6
    if use_vocal:
        v_f0, v_f1, v_f2, v_f3 = _vocal_formants(p104)
        v_ton = float(p104.get("tonality") or 0.6)
    if dur is None:
        dur = max(s.birth + s.lifetime for s in specs) + 0.05
    n = int(dur * sr) + 2048
    out = np.zeros(n, dtype=np.float64)

    if mode == "saw":
        saw = _saw_from_specs(specs, n, sr, meta=meta)
        fade = max(2, int(FADE_MS * 1e-3 * sr))
        if fade * 2 <= len(saw):
            w = _hann(fade * 2)
            saw[:fade] *= w[:fade]
            saw[-fade:] *= w[fade:]
        out[: len(saw)] += saw * 0.35
    elif mode == "am" and len(specs) == 1:
        spec = specs[0]
        md = max(spec.mod_depth, parent_mod, float(p104.get("mod_depth") or 0))
        mr = spec.mod_rate or parent_mrate or float(p104.get("mod_rate") or 0)
        grain = _render_am_mode(spec, sr, mod_depth=md, mod_rate=mr, meta=meta)
        s0 = int(spec.birth * sr)
        e = min(len(grain), n - s0)
        if e > 0 and s0 >= 0:
            out[s0:s0 + e] += grain[:e]
    elif mode == "fm" and len(specs) == 1:
        spec = specs[0]
        rng = np.random.default_rng(spec.seed)
        grain = _render_fm_mode(spec, sr, rng, parent_hr=parent_hr)
        s0 = int(spec.birth * sr)
        e = min(len(grain), n - s0)
        if e > 0 and s0 >= 0:
            out[s0:s0 + e] += grain[:e]
    else:
        single = len(specs) == 1
        komar = _komar_profile(specs, parent_hr)
        komar_am = (max(parent_mod, 0.22), parent_mrate if parent_mrate > 0 else 6.2)
        nb_default = 0.15 if single and parent_hr < 0.35 and specs[0].фаза == "тон" else 0.0
        for i, spec in enumerate(specs):
            rng = np.random.default_rng(spec.seed)
            md = max(spec.mod_depth, parent_mod) if single else spec.mod_depth
            mr = spec.mod_rate or parent_mrate if single else spec.mod_rate
            spec = _dozhd_sustain_grain(spec, specs, parent_hr, dur)
            spec = _heartbeat_stretch_grain(spec, specs, parent_hr)
            spec = _frog_croak_grain(spec, specs, parent_hr)
            spec, wb = _wideband_kwargs(spec, parent_hr, single, meta)
            bs = wb.get("band_scale", 1.0)
            nb = wb.get("noise_boost", nb_default)
            if buk_cls == "шипящая" or vflag == "noise":
                nb = max(nb, 0.24)
                bs = max(bs, 1.14)
            if (
                vflag == "noise"
                and _буква_буквицы(meta) in _ВЗРЫВНЫЕ
                and parent_hr < 0.08
            ):
                nb = max(nb, 0.30)
                bs = max(bs, 1.10)
            amp = _hi_atten(spec, impulsive)
            if (
                buk_cls == "согласная"
                and _буква_буквицы(meta) in _ВЗРЫВНЫЕ
                and parent_hr < 0.15
            ):
                amp *= 1.12
                nb = max(nb, 0.16)
            if (
                buk_cls == "согласная"
                and _буква_буквицы(meta) in _ЗВОНКИЕ_ВЗРЫВ
                and parent_hr < 0.12
                and not use_vocal
            ):
                nb = max(nb, 0.22)
                amp *= 1.06
            grain_buzz = buzz
            if komar:
                whine = _komar_whine_render(spec)
                if whine:
                    amp, grain_buzz = whine
            if use_vocal:
                s0, grain_v = _render_vocal_grain(
                    spec, sr, rng,
                    f0p=v_f0, f1=v_f1, f2=v_f2, f3=v_f3, tonality=v_ton,
                    profile=vprof,
                )
                if vocal_blend:
                    s0g, grain_g = _render_gen_grain(
                        spec, sr, rng,
                        partial=None,
                        mod_depth=md,
                        mod_rate=mr,
                        noise_boost=nb,
                        band_scale=bs,
                        amp_scale=amp,
                        buzz=grain_buzz,
                    )
                    m = max(len(grain_v), len(grain_g))
                    gv = np.zeros(m); gv[: len(grain_v)] = grain_v
                    gg = np.zeros(m); gg[: len(grain_g)] = grain_g
                    grain = vocal_vocal_w * gv + (1.0 - vocal_vocal_w) * gg
                    s0 = min(s0, s0g)
                else:
                    grain = grain_v
            else:
                if buk_cls == "шипящая" and not use_vocal:
                    s0, grain = _render_fricative_grain(
                        spec, sr, rng, p104, amp_scale=amp,
                    )
                else:
                    s0, grain = _render_gen_grain(
                        spec, sr, rng,
                        partial=None,
                        mod_depth=md,
                        mod_rate=mr,
                        noise_boost=nb,
                        band_scale=bs,
                        amp_scale=amp,
                        buzz=grain_buzz,
                    )
            e = min(len(grain), n - s0)
            if e > 0 and s0 >= 0:
                out[s0:s0 + e] += grain[:e]

        if komar:
            out = связать_полосы(out, sr, 0.11)
            am_d, am_r = komar_am
            if am_d > 0.01 and am_r > 0:
                t = np.arange(len(out), dtype=np.float64) / sr
                out *= 1.0 + am_d * np.sin(2.0 * np.pi * am_r * t)

        if _heartbeat_pulse(specs, parent_hr):
            t = np.arange(n, dtype=np.float64) / sr
            out *= 1.0 + 0.5 * np.sin(2.0 * np.pi * 1.2 * t)

    if mode in ("saw", "am", "fm"):
        out = _trim_tail_pad(out, sr)
    m = np.max(np.abs(out))
    return (out / m * 0.9 if m > 0 else out).astype(np.float64)


def синтез_из_json_rec(rec: dict, *, sr: int = SR_DEFAULT) -> tuple[np.ndarray, int]:
    atoms = rec.get("atoms") or []
    crosses = rec.get("crosses")
    dur = rec.get("длительность_сек")
    y = синтез_из_атомов(atoms, crosses, sr=sr, dur=float(dur) + 0.1 if dur else None, meta=rec)
    return y, sr


# ── per-band envelope coupling (stitch v2, MVP) ─────────────────────────────
# НИЧЕГО ВЫШЕ НЕ МЕНЯЕТСЯ. Это отдельная, opt-in ступень: берёт уже готовый
# FULL-сигнал и СВЯЗЫВАЕТ огибающие октавных полос к целевой вложенности.
# Та же сетка полос, что и в оси.вложенность() — связываем ровно то, что метрика
# и меряет (со-модуляцию огибающих полос), а не общий глобальный синус.
NEST_BANDS = [(120, 240), (240, 480), (480, 960),
              (960, 1920), (1920, 3840), (3840, 7500)]


def _band_split(y: np.ndarray, sr: int):
    """Раскладывает сигнал на октавные полосы (та же сетка, что у метрики)."""
    from scipy import signal as _sig
    bands, idx = [], []
    for k, (lo, hi) in enumerate(NEST_BANDS):
        hi = min(hi, sr / 2 - 1)
        if hi <= lo:
            continue
        sos = _sig.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        bands.append(_sig.sosfiltfilt(sos, y))
        idx.append(k)
    return bands, idx


def _band_envelope(bp: np.ndarray, sr: int):
    from scipy import signal as _sig
    e = np.abs(_sig.hilbert(bp))
    if len(e) > 205:
        e = _sig.savgol_filter(e, 201, 2)
    return np.maximum(e, 1e-6)


def связать_полосы(y: np.ndarray, sr: int, target: float, *,
                   max_gain: float = 4.0) -> np.ndarray:
    """stitch v2 (MVP). Связывает огибающие октавных полос к целевой
    со-модуляции `target` (≈ params104.nestedness живой буквы).

    Идея: метрика nestedness = средняя корреляция огибающих полос. Глобальный
    shared_env (один синус) её не двигает (см. диагностику). Здесь каждая полоса
    получает СВОЮ огибающую = смесь собственной и общей формы:
        env'_b = (1-α)·own_b + α·(common · среднее(own_b))
    где common — общая относительная форма (среднее нормированных огибающих),
    α = clamp(target). При α→1 полосы синхронны (corr→1), при α→0 — как есть.
    FULL-путь не затрагивается; это пост-стадия над готовым y.
    """
    if y is None or len(y) < 64:
        return y
    α = float(min(0.95, max(0.0, target)))
    if α <= 0.01:
        return y
    bands, _ = _band_split(y, sr)
    if len(bands) < 2:
        return y
    envs = [_band_envelope(b, sr) for b in bands]
    n = min(len(e) for e in envs)
    envs = [e[:n] for e in envs]
    bands = [b[:n] for b in bands]
    # общая относительная форма (масштаб-независимая): среднее по полосам
    rel = np.mean([e / (e.mean() + 1e-9) for e in envs], axis=0)
    out = np.zeros(n, dtype=np.float64)
    for b, e in zip(bands, envs):
        target_env = (1 - α) * e + α * (rel * (e.mean() + 1e-9))
        gain = np.clip(target_env / e, 1.0 / max_gain, max_gain)
        out += b * gain
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def синтез_per_band(
    atoms: list[dict],
    crosses: list[dict] | None = None,
    *,
    sr: int = SR_DEFAULT,
    dur: float | None = None,
    meta: dict | None = None,
    max_atoms: int = 8000,
    target_nest: float | None = None,
) -> np.ndarray:
    """Полный путь stitch v2 = FULL-синтез + связывание полос.
    target_nest по умолчанию берётся из params104.nestedness (цель — живое).
    """
    y = синтез_из_атомов(atoms, crosses, sr=sr, dur=dur, meta=meta, max_atoms=max_atoms)
    if target_nest is None:
        p104 = (meta or {}).get("параметры104") or (meta or {}).get("parent_params_full") or {}
        target_nest = float(p104.get("nestedness") or (meta or {}).get("nestedness") or 0.0)
    return связать_полосы(y, sr, target_nest)

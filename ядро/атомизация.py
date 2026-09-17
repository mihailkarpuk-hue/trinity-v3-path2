# -*- coding: utf-8 -*-
"""Атомизация V2: пики → треки → атомы (закрытый трек = атом).

SR=22050, N=2048, HOP=512. Связь соседних кадров: |log2(f_new/f_old)| ≤ 1/12,
разрыв ≤ 1 кадр, жадно по ближайшей частоте. Трек без продолжения ≥ 2 кадров → закрыт.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ядро.фаза import hps_harm, hps_pitch
from ядро.пороги import (
    BAND_ENERGY_MIN,
    BAND_F_HI,
    BAND_F_LO,
    BAND_N,
    CLOSE_IDLE_FRAMES,
    FLATNESS_ШУМ,
    FLATNESS_ТОН_ЧИСТЫЙ,
    HARMONICITY_ПИК_СМЕШАН,
    HOP,
    LOG2_FREQ_TOL,
    MAX_FRAME_GAP,
    N_FFT,
    PEAK_AMP_MIN,
    SR,
)


@dataclass
class _Peak:
    frame: int
    time: float
    freq: float
    amp: float
    harmonicity: float
    kind: str  # "ton" | "shum"


@dataclass
class _Track:
    peaks: list[_Peak] = field(default_factory=list)
    idle: int = 0

    @property
    def last(self) -> _Peak:
        return self.peaks[-1]


def _hann(n: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


def _flatness(mag: np.ndarray) -> float:
    nz = mag[mag > 1e-9]
    if len(nz) == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(nz))) / (nz.mean() + 1e-12))


def _peaks_кадра(mag: np.ndarray, fr: np.ndarray, frame: int, t: float) -> list[_Peak]:
    flat = _flatness(mag)
    out: list[_Peak] = []

    if flat > FLATNESS_ШУМ:
        edges = np.logspace(math.log10(BAND_F_LO), math.log10(BAND_F_HI), BAND_N + 1)
        for b in range(BAND_N):
            m = (fr >= edges[b]) & (fr < edges[b + 1])
            e = float(mag[m].sum())
            if e > BAND_ENERGY_MIN:
                f = float(math.sqrt(edges[b] * edges[b + 1]))
                out.append(_Peak(frame, t, f, min(1.0, e * 6.0), 0.0, "shum"))
        if not out:
            et = float(mag.sum())
            if et > 0.005:
                cen = float((fr * mag).sum() / (et + 1e-12))
                out.append(_Peak(frame, t, cen, min(1.0, float(mag.max()) * 6.0), 0.0, "shum"))
    else:
        harm = hps_harm(mag, SR, N_FFT)
        h = 1.0 if flat < FLATNESS_ТОН_ЧИСТЫЙ else HARMONICITY_ПИК_СМЕШАН
        best_k, best_v = -1, PEAK_AMP_MIN
        for k in range(2, len(mag) - 1):
            if mag[k] > best_v and mag[k] > mag[k - 1] and mag[k] > mag[k + 1]:
                if 60 <= fr[k] <= 9000:
                    best_k, best_v = k, float(mag[k])
        if best_k >= 0:
            out.append(_Peak(frame, t, float(fr[best_k]), min(1.0, best_v * 6.0), h, "ton"))
        elif harm > 0.3:
            pitch = hps_pitch(mag, SR, N_FFT)
            if pitch > 0:
                pk = float(mag.max()) if len(mag) else 0.0
                out.append(_Peak(frame, t, pitch, min(1.0, pk * 6.0), h, "ton"))
    return out


def _log2_ratio_ok(f1: float, f2: float) -> bool:
    if f1 <= 0 or f2 <= 0:
        return False
    return abs(math.log2(f2 / f1)) <= LOG2_FREQ_TOL


def _link_tracks(tracks: list[_Track], peaks: list[_Peak], frame: int) -> None:
    used: set[int] = set()
    candidates: list[tuple[float, int, int, bool]] = []
    for ti, tr in enumerate(tracks):
        if tr.idle > MAX_FRAME_GAP:
            continue
        gap = frame - tr.last.frame
        if gap < 1 or gap > MAX_FRAME_GAP + 1:
            continue
        for pi, pk in enumerate(peaks):
            ok = _log2_ratio_ok(tr.last.freq, pk.freq)
            d = abs(math.log2(pk.freq / tr.last.freq)) if tr.last.freq > 0 and pk.freq > 0 else 999.0
            candidates.append((d, ti, pi, ok))

    # сначала связи в пределах полутона, затем монорidge (1 пик, 1 трек)
    candidates.sort(key=lambda x: (not x[3], x[0]))
    matched_tracks: set[int] = set()
    for d, ti, pi, ok in candidates:
        if ti in matched_tracks or pi in used:
            continue
        if not ok:
            if len(peaks) != 1 or len([t for t in tracks if t.idle <= MAX_FRAME_GAP]) != 1:
                continue
        tracks[ti].peaks.append(peaks[pi])
        tracks[ti].idle = 0
        matched_tracks.add(ti)
        used.add(pi)

    for ti, tr in enumerate(tracks):
        if ti not in matched_tracks:
            tr.idle += 1

    for pi, pk in enumerate(peaks):
        if pi not in used:
            tracks.append(_Track(peaks=[pk], idle=0))


def _close_idle(tracks: list[_Track]) -> tuple[list[_Track], list[_Track]]:
    active, closed = [], []
    for tr in tracks:
        if tr.idle >= CLOSE_IDLE_FRAMES:
            closed.append(tr)
        else:
            active.append(tr)
    return active, closed


def _downsample(ts: list[float], vs: list[float], max_pts: int = 32) -> tuple[list[float], list[float]]:
    if len(ts) <= max_pts:
        return ts, vs
    idx = np.linspace(0, len(ts) - 1, max_pts).astype(int)
    return [ts[i] for i in idx], [vs[i] for i in idx]


def _freq_slope(ts: list[float], freqs: list[float]) -> float:
    if len(ts) < 2:
        return 0.0
    t = np.asarray(ts, dtype=np.float64)
    y = np.log2(np.maximum(np.asarray(freqs, dtype=np.float64), 1.0))
    dt = t[-1] - t[0]
    if dt <= 1e-9:
        return 0.0
    slope = float(np.polyfit(t - t[0], y, 1)[0])
    return slope


def _attack_ratio(ts: list[float], amps: list[float]) -> float:
    if len(amps) < 1:
        return 0.0
    a = np.asarray(amps, dtype=np.float64)
    t = np.asarray(ts, dtype=np.float64)
    i_peak = int(np.argmax(a))
    life = t[-1] - t[0] + (HOP / SR)
    if life <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, (t[i_peak] - t[0]) / life)))


def _decay_shape(ts: list[float], amps: list[float]) -> float:
    if len(amps) < 3:
        return 0.0
    a = np.asarray(amps, dtype=np.float64)
    t = np.asarray(ts, dtype=np.float64)
    i_peak = int(np.argmax(a))
    if i_peak >= len(a) - 2:
        return 0.0
    tail_t = t[i_peak:] - t[i_peak]
    tail_a = np.maximum(a[i_peak:], 1e-9)
    if tail_t[-1] <= 1e-9:
        return 0.0
    log_a = np.log(tail_a)
    slope = float(np.polyfit(tail_t, log_a, 1)[0])
    return slope


def _track_to_atom(tr: _Track) -> dict:
    ts = [p.time for p in tr.peaks]
    freqs = [p.freq for p in tr.peaks]
    amps = [p.amp for p in tr.peaks]
    harms = [p.harmonicity for p in tr.peaks]
    birth = ts[0]
    lifetime = ts[-1] - ts[0] + (HOP / SR)
    ft, ff = _downsample(ts, freqs)
    _, fa = _downsample(ts, amps)
    kind = tr.peaks[0].kind
    harm_med = float(np.median(harms))
    return {
        "birth": round(birth, 4),
        "lifetime": round(lifetime, 4),
        "freq": round(float(np.median(freqs)), 2),
        "amp": round(float(np.median(amps)), 4),
        "harmonicity": round(harm_med, 4),
        "harmonic_index": 1 if kind == "ton" and harm_med > 0.5 else 0,
        "phase": harm_med,  # числовой legacy; строковая фаза — ниже
        "freq_slope": round(_freq_slope(ts, freqs), 4),
        "attack_ratio": round(_attack_ratio(ts, amps), 4),
        "decay_shape": round(_decay_shape(ts, amps), 4),
        "freq_t": [round(x, 4) for x in ff],
        "amp_t": [round(x, 4) for x in fa],
        "_kind": kind,
    }


def _phase_str(h: float) -> str:
    from ядро.фаза import фаза
    return фаза(h)


def атомизировать(x: np.ndarray, sr: int = SR) -> list[dict]:
    """Звук → список атомов (закрытые треки)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
        sr = SR

    win = _hann(N_FFT)
    fr = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    tracks: list[_Track] = []
    closed_atoms: list[dict] = []

    frame = 0
    starts = list(range(0, max(1, len(x) - N_FFT + 1), HOP))
    if not starts:
        starts = [0]
    for i in starts:
        seg = x[i : i + N_FFT]
        if len(seg) < N_FFT:
            seg = np.pad(seg, (0, N_FFT - len(seg)))
        mag = np.abs(np.fft.rfft(seg * win)) / (N_FFT / 2)
        t = i / SR
        peaks = _peaks_кадра(mag, fr, frame, t)
        if peaks:
            _link_tracks(tracks, peaks, frame)
        else:
            for tr in tracks:
                tr.idle += 1
        tracks, closed = _close_idle(tracks)
        for tr in closed:
            if len(tr.peaks) >= 1:
                closed_atoms.append(_track_to_atom(tr))
        frame += 1

    for tr in tracks:
        if tr.peaks:
            closed_atoms.append(_track_to_atom(tr))

    # строковая фаза + сортировка по birth
    for a in closed_atoms:
        a["phase"] = _phase_str(float(a.get("harmonicity") or 0))
    closed_atoms.sort(key=lambda a: a["birth"])
    return closed_atoms


def атомизировать_синтетику(
    kind: str,
    *,
    sr: int = SR,
    dur: float = 0.2,
) -> np.ndarray:
    """Генераторы для tests/test_атомизация.py."""
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float64) / sr
    if kind == "chirp":
        f0, f1 = 400.0, 800.0
        phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t ** 2)
        return np.sin(phase).astype(np.float64)
    if kind == "tone":
        return np.sin(2 * np.pi * 440.0 * t).astype(np.float64)
    if kind == "click":
        n = int(0.1 * sr)
        x = np.zeros(n, dtype=np.float64)
        x[: max(1, int(0.002 * sr))] = 1.0
        return x
    raise ValueError(kind)

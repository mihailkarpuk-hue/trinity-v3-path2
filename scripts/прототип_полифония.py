# -*- coding: utf-8 -*-
"""ПРОТОТИП полифонной тональной атомизации на 1 клетке (живая_О).
Ядро НЕ трогаем. Здесь автономный атомайзер: гейт тишины + top-N пиков/кадр
(вместо 1) → гласная = несколько формантных тон-атомов. Сравниваем ε (один
прокси) для: stored(21 шум) vs полифон-N. Только чтение.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ядро.атомизация import (  # noqa
    _hann, _flatness, _link_tracks, _close_idle, _track_to_atom, _phase_str, _Peak,
)
from ядро.фаза import hps_harm, hps_pitch  # noqa
from ядро.пороги import (  # noqa
    N_FFT, HOP, SR, FLATNESS_ШУМ, FLATNESS_ТОН_ЧИСТЫЙ, HARMONICITY_ПИК_СМЕШАН,
    PEAK_AMP_MIN, BAND_ENERGY_MIN, BAND_F_LO, BAND_F_HI, BAND_N,
)
from ядро.метрика import d9, d9_по_группам, вектор_из_словаря, загрузить_индекс  # noqa
from ядро.синтез import синтезировать  # noqa
import ядро.словарь_кирпичей as СЛ  # noqa
import добыча_словаря as Д  # noqa

ТИШИНА_ОТН = 0.0224  # из scripts/вывести_порог_тишины.py


def _пики_тон_топN(mag, fr, frame, t, N):
    """top-N локальных максимумов тональной ветки."""
    flat = _flatness(mag)
    out = []
    if flat > FLATNESS_ШУМ:
        edges = np.logspace(math.log10(BAND_F_LO), math.log10(BAND_F_HI), BAND_N + 1)
        for b in range(BAND_N):
            m = (fr >= edges[b]) & (fr < edges[b + 1])
            e = float(mag[m].sum())
            if e > BAND_ENERGY_MIN:
                f = float(math.sqrt(edges[b] * edges[b + 1]))
                out.append(_Peak(frame, t, f, min(1.0, e * 6.0), 0.0, "shum"))
        return out
    h = 1.0 if flat < FLATNESS_ТОН_ЧИСТЫЙ else HARMONICITY_ПИК_СМЕШАН
    cand = []
    for k in range(2, len(mag) - 1):
        if mag[k] > PEAK_AMP_MIN and mag[k] > mag[k - 1] and mag[k] > mag[k + 1]:
            if 60 <= fr[k] <= 9000:
                cand.append((float(mag[k]), float(fr[k])))
    cand.sort(reverse=True)
    for v, f in cand[:N]:
        out.append(_Peak(frame, t, f, min(1.0, v * 6.0), h, "ton"))
    if not out and flat <= FLATNESS_ТОН_ЧИСТЫЙ:
        harm = hps_harm(mag, SR, N_FFT)
        if harm > 0.3:
            p = hps_pitch(mag, SR, N_FFT)
            if p > 0:
                out.append(_Peak(frame, t, p, min(1.0, float(mag.max()) * 6.0), h, "ton"))
    return out


def атомизировать_полифон(x, sr, N=6):
    x = np.asarray(x, float)
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m
    win = _hann(N_FFT)
    fr = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    starts = list(range(0, max(1, len(x) - N_FFT + 1), HOP)) or [0]
    mags = []
    for i in starts:
        seg = x[i:i + N_FFT]
        if len(seg) < N_FFT:
            seg = np.pad(seg, (0, N_FFT - len(seg)))
        mags.append(np.abs(np.fft.rfft(seg * win)) / (N_FFT / 2))
    порог_тихо = ТИШИНА_ОТН * max((float(mm.max()) for mm in mags), default=0.0)
    tracks, closed_atoms = [], []
    for frame, i in enumerate(starts):
        mag = mags[frame]
        peaks = [] if float(mag.max()) < порог_тихо else _пики_тон_топN(mag, fr, frame, i / SR, N)
        if peaks:
            _link_tracks(tracks, peaks, frame)
        else:
            for tr in tracks:
                tr.idle += 1
        tracks, closed = _close_idle(tracks)
        for tr in closed:
            if len(tr.peaks) >= 1:
                closed_atoms.append(_track_to_atom(tr))
    for tr in tracks:
        if tr.peaks:
            closed_atoms.append(_track_to_atom(tr))
    for a in closed_atoms:
        a["phase"] = _phase_str(float(a.get("harmonicity") or 0))
    closed_atoms.sort(key=lambda a: a["birth"])
    return [{k: v for k, v in a.items() if not k.startswith("_")} for a in closed_atoms]


def eps_прокси(cid, atoms, словарь, idx):
    c = Д._cell(cid)
    x, sr = Д._load_audio(Д._audio_path(c))
    before = Д._фенотип(x, sr)
    gens, dur, состав = СЛ.собрать_гены_из_атомов(atoms, словарь, labels_map=None, cid=cid, blend=True)
    y = синтезировать(gens, sr=Д.SR, dur=dur, meta=c)
    after = Д._фенотип(y, Д.SR)
    va, vb = вектор_из_словаря(before), вектор_из_словаря(after)
    eps = d9(va, vb, mu=idx["mu"], sd=idx["sd"])
    from collections import Counter
    ph = Counter(a.get("phase") for a in atoms)
    top = list(состав.items())[:3]
    print(f"  атомов={len(atoms):3d} фазы={dict(ph)} ε={eps:.2f}")
    print(f"    состав: {', '.join(f'{k} {v*100:.0f}%' for k,v in top)}")
    гр = d9_по_группам(va, vb, mu=idx["mu"], sd=idx["sd"])
    print(f"    SPECTRAL={гр['SPECTRAL']:.1f}")
    return eps


def main():
    cid = sys.argv[1] if len(sys.argv) > 1 else "живая_О"
    словарь = СЛ.загрузить()
    idx = загрузить_индекс()
    c = Д._cell(cid)
    jp = os.path.join(Д.КЛЕТКИ, c.get("атомы") or "")
    stored = json.load(open(jp, encoding="utf-8")).get("atoms") or []
    x, sr = Д._load_audio(Д._audio_path(c))
    print(f"=== {cid} (прокси: labels_map=None, meta=клетка) ===")
    print("[stored (baseline)]")
    eps_прокси(cid, stored, словарь, idx)
    for N in (4, 6, 8, 12):
        print(f"[полифон N={N} + гейт]")
        poly = атомизировать_полифон(x, sr, N=N)
        eps_прокси(cid, poly, словарь, idx)


if __name__ == "__main__":
    main()

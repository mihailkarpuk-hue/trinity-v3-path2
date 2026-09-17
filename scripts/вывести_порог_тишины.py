# -*- coding: utf-8 -*-
"""Вывод порога гейта тишины ИЗ КОРПУСА (канон п.12: константа с происхождением).

Для каждого звука корпуса: нормируем к пику 1.0 (как атомизатор), считаем
пик магнитуды каждого кадра, берём отношение к максимальному кадру записи в дБ.
Пул по всему корпусу → гистограмма → впадина между «тишина» и «сигнал».
Только чтение; печатает число-кандидат.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import добыча_словаря as Д  # noqa
from ядро.атомизация import _hann  # noqa
from ядро.пороги import N_FFT, HOP, SR  # noqa


def кадровые_пики(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m
    win = _hann(N_FFT)
    peaks = []
    for i in range(0, max(1, len(x) - N_FFT + 1), HOP):
        seg = x[i:i + N_FFT]
        if len(seg) < N_FFT:
            seg = np.pad(seg, (0, N_FFT - len(seg)))
        mag = np.abs(np.fft.rfft(seg * win)) / (N_FFT / 2)
        peaks.append(float(mag.max()))
    return np.asarray(peaks) if peaks else np.zeros(1)


def main() -> None:
    ids = Д.id_корпуса()
    все_дб = []
    per_type = {"базис": [], "буквица": []}
    for cid in ids:
        c = Д._cell(cid)
        ap = Д._audio_path(c)
        if not ap:
            continue
        try:
            x, sr = Д._load_audio(ap)
        except Exception:
            continue
        pk = кадровые_пики(x)
        mx = pk.max()
        if mx <= 0:
            continue
        дб = 20.0 * np.log10(np.clip(pk / mx, 1e-9, None))
        все_дб.append(дб)
        grp = "буквица" if c.get("группа") == "буквица_живая" else "базис"
        per_type[grp].append(дб)

    all_db = np.concatenate(все_дб)
    print(f"корпус: {len(все_дб)} записей, {len(all_db)} кадров")
    # гистограмма по дБ от -100 до 0
    bins = np.arange(-100, 1, 2.0)
    hist, edges = np.histogram(all_db, bins=bins)
    print("\nгистограмма пик-кадра (дБ отн. макс. кадра записи):")
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"  [{e0:6.0f}..{e1:6.0f}] {h:5d} {'█'*int(40*h/max(hist.max(),1))}")

    # впадина: минимум гистограммы между модой тишины (низ) и модой сигнала (верх)
    # ищем локальный минимум в диапазоне [-70, -20]
    lo_i = np.searchsorted(edges, -70)
    hi_i = np.searchsorted(edges, -18)
    окно = hist[lo_i:hi_i]
    if len(окно):
        valley_rel = int(np.argmin(окно))
        valley_db = 0.5 * (edges[lo_i + valley_rel] + edges[lo_i + valley_rel + 1])
    else:
        valley_db = -40.0
    print(f"\nвпадина (тишина↔сигнал) ≈ {valley_db:.1f} дБ")
    print(f"→ порог линейный τ = {10**(valley_db/20):.5f} (доля от пика кадра записи)")
    # доля кадров ниже впадины по типам
    for grp, arrs in per_type.items():
        if not arrs:
            continue
        d = np.concatenate(arrs)
        print(f"  {grp:9s}: {100*np.mean(d < valley_db):.0f}% кадров ниже впадины (тишина)")


if __name__ == "__main__":
    main()

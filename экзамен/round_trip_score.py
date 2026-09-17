# -*- coding: utf-8 -*-
"""ЭКЗАМЕН · Контракт обратимости: звук → атомы → звук, с измеримой потерей.
Двойная приёмка (из находок):
  тональное — по спектру формы волны (корреляция STFT-магнитуд);
  шумовое   — по спектральному характеру (корреляция средних спектров).
Это доказательство, что структура несёт информацию, а не украшает.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ядро'))
from фаза import загрузить, отпечаток  # один канон фазы

N = 2048; HOP = 512

def stft_mag(x, sr):
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    return np.array([np.abs(np.fft.rfft(x[i:i + N] * win)) for i in range(0, max(1, len(x) - N), HOP)])

def band_spectrogram(x, sr, nb=40):
    """Полосовая спектрограмма (время × 40 лог-полос), лог-сжатая.
    Несёт ВРЕМЕННУЮ структуру: когда энергия появляется в какой полосе —
    нужный критерий для шума (сибилянты, импульсы), где суть во времени."""
    S = stft_mag(x, sr); fr = np.fft.rfftfreq(N, 1 / sr)
    edges = np.logspace(np.log10(40), np.log10(min(sr / 2, 11000)), nb + 1)
    out = np.zeros((len(S), nb))
    for b in range(nb):
        m = (fr >= edges[b]) & (fr < edges[b + 1])
        if m.any(): out[:, b] = S[:, m].sum(1)
    return np.log1p(out)

# ── прямой ход: звук → атомы (пики спектра по кадрам) ──
def атомизировать(x, sr):
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1))); fr = np.fft.rfftfreq(N, 1 / sr)
    atoms = []
    for i in range(0, max(1, len(x) - N), HOP):
        mag = np.abs(np.fft.rfft(x[i:i + N] * win)) / (N / 2)
        nz = mag[mag > 1e-9]; flat = float(np.exp(np.log(nz).mean()) / (nz.mean() + 1e-12)) if len(nz) else 1.0
        t = i / sr
        if flat > 0.4:  # шумовой кадр — ПЛОТНАЯ реконструкция спектральной огибающей (32 лог-полосы)
            edges = np.logspace(np.log10(40), np.log10(min(sr / 2, 11000)), 33)
            for b in range(32):
                lo, hi = edges[b], edges[b + 1]; m = (fr >= lo) & (fr < hi)
                e = float(mag[m].sum())
                if e > 0.0008: atoms.append((t, float(np.sqrt(lo * hi)), float(min(1, e * 6)), 0.0))
        else:  # тональные пики
            for k in range(2, len(mag) - 1):
                if mag[k] > 0.004 and mag[k] > mag[k - 1] and mag[k] > mag[k + 1]:
                    atoms.append((t, float(fr[k]), float(min(1, mag[k] * 6)), 0.9))
    return atoms

# ── обратный ход: атомы → звук (гранулярный синтез) ──
def синтез(atoms, sr, dur):
    n = int(dur * sr) + N; out = np.zeros(n); gl = int(0.06 * sr)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(gl) / (gl - 1)))
    for (t, f, a, h) in atoms:
        if f <= 20 or f >= sr / 2 or a <= 0: continue
        s = int(t * sr); w = 2 * np.pi * f / sr; idx = np.arange(gl)
        # аддитивный ресинтез: тон → когерентная фаза; шум → случайная фаза на той же частоте.
        # Плотная сумма случайно-фазовых частиц воспроизводит спектральную огибающую шума.
        phase = 0.0 if h > 0.5 else np.random.rand() * 2 * np.pi
        out[s:s + gl] += a * 0.28 * env * np.sin(w * idx + phase)
    return out[:int(dur * sr)]

def corr(a, b):
    a = a.ravel(); b = b.ravel(); m = min(len(a), len(b)); a, b = a[:m], b[:m]
    if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def score(path):
    x, sr = загрузить(path); dur = len(x) / sr
    fp = отпечаток(x, sr); dom = fp['доминанта']
    atoms = атомизировать(x, sr); y = синтез(atoms, sr, dur)
    if dom == 'тон':                              # форма волны: корреляция STFT-магнитуд
        A, B = stft_mag(x, sr), stft_mag(y, sr); m = min(len(A), len(B)); s = corr(A[:m], B[:m]); crit = 'specsim(STFT)'
    else:                                          # шум/переход: ВРЕМЕННОЙ критерий — полосовая спектрограмма
        A = band_spectrogram(x, sr); B = band_spectrogram(y, sr); m = min(len(A), len(B)); s = corr(A[:m], B[:m]); crit = 'врем-спектрограмма'
    return dict(звук=os.path.basename(path), фаза=dom, критерий=crit, score=round(max(0, s), 3),
                атомов=len(atoms), отпечаток=fp)

if __name__ == '__main__':
    ET = os.path.join(os.path.dirname(__file__), '..', 'данные', 'клеточки', 'эталоны')
    args = sys.argv[1:] or sorted(glob.glob(os.path.join(ET, '*_real.wav')))
    print(f"{'звук':<26}{'фаза':<9}{'критерий':<18}{'score':>6}  атомов")
    tot = []
    for p in args:
        try:
            r = score(p); tot.append(r['score'])
            print(f"{r['звук'][:25]:<26}{r['фаза']:<9}{r['критерий']:<18}{r['score']:>6}  {r['атомов']}")
        except Exception as e:
            print(os.path.basename(p), 'ошибка', e)
    if tot: print(f"\nсредний round-trip score: {round(sum(tot)/len(tot),3)}  (n={len(tot)})")

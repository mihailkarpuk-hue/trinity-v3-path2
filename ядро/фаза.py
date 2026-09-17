# -*- coding: utf-8 -*-
"""ЯДРО · Фаза звукового вещества (диаграмма Гиббса).
Единственный канон фазы для всего V3: flatness-гейт + HPS harmonic_ratio,
как в scripts/atoms_full103.py. Чистая физика, без UI.
Фаза ∈ {тон, шум, переход}. Пороги: тон>0.6 · шум<0.05 · переход 0.05–0.6.
"""
import numpy as np, wave

def загрузить(path):
    with wave.open(path) as w:
        sr = w.getframerate(); ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    return (x.reshape(-1, 2).mean(1) if ch == 2 else x), sr

def hps_pitch(mag, sr, N):
    """Оценка f0 (Гц) через HPS; 0 если не найден."""
    bh = sr / N
    mn = max(2, int(60 / bh))
    mx = min(len(mag) - 1, int(3000 / bh))
    if mx <= mn:
        return 0.0
    hps = mag[mn:mx + 1].copy()
    for h in (2, 3, 4):
        end = min(len(mag), (mx + 1) * h)
        dec = mag[mn * h:end:h]
        n = min(len(hps), len(dec))
        hps[:n] *= dec[:n]
    best = int(np.argmax(hps)) + mn
    half = best // 2
    if half >= mn and mag[half] > 0.4 * mag[best]:
        best = half
    pitch = best * bh
    return float(pitch) if pitch >= 60 else 0.0


def hps_harm(mag, sr, N):
    bh = sr / N; mn = max(2, int(60 / bh)); mx = min(len(mag) - 1, int(3000 / bh))
    if mx <= mn: return 0.0
    hps = mag[mn:mx + 1].copy()
    for h in (2, 3, 4):
        end = min(len(mag), (mx + 1) * h); dec = mag[mn * h:end:h]
        n = min(len(hps), len(dec)); hps[:n] *= dec[:n]
    best = int(np.argmax(hps)) + mn; half = best // 2
    if half >= mn and mag[half] > 0.4 * mag[best]: best = half
    pitch = best * bh
    if pitch < 60: return 0.0
    et = float(mag.sum() + 1e-9)
    eh = sum(float(mag[int(round(pitch * k / bh))]) for k in range(1, 8) if int(round(pitch * k / bh)) < len(mag))
    return min(1.0, eh / et * 6)

def harmonicity_по_кадрам(x, sr, N=2048):
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1))); H = []
    for i in range(0, max(1, len(x) - N), int(sr * 0.016)):
        mag = np.abs(np.fft.rfft(x[i:i + N] * win)); nz = mag[mag > 1e-9]
        flat = float(np.exp(np.mean(np.log(nz))) / (nz.mean() + 1e-12)) if len(nz) else 1.0
        H.append(0.0 if flat > 0.3 else hps_harm(mag, sr, N))
    return np.array(H)

def фаза(h):
    return 'тон' if h > 0.6 else ('шум' if h < 0.05 else 'переход')

def отпечаток(x, sr):
    """Фазовый отпечаток звука: доли тон/шум/переход + доминанта."""
    H = harmonicity_по_кадрам(x, sr)
    if not len(H): return dict(тон=0.0, шум=100.0, переход=0.0, доминанта='шум')
    t = float((H > 0.6).mean() * 100); n = float((H < 0.05).mean() * 100); tr = 100 - t - n
    dom = 'тон' if t >= max(n, tr) else ('шум' if n >= tr else 'переход')
    return dict(тон=round(t, 1), шум=round(n, 1), переход=round(tr, 1), доминанта=dom)

def паспорт(x, sr, полный=True):
    """ШОВ 1 · ЕДИНЫЙ паспорт звука за один проход (одно тело — один анализатор).
    Возвращает: фаза + гармоничность + наши оси (FD/R²/вложенность/модуляция),
    и при полный=True — полные 104 параметра (параметры104 = 103 + 5 осей).
    На уровне КЛЕТКИ — полный файл; per-atom params_104 — обогатить_атомы_104.py."""
    import os, sys
    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in sys.path:
        sys.path.insert(0, _d)
    from оси import оси_звука

    H = harmonicity_по_кадрам(x, sr)
    harm = round(float(H.mean()), 3) if len(H) else 0.0
    fp = отпечаток(x, sr)
    оси = оси_звука(x, sr)
    p = {
        "phase": fp["доминанта"],
        "harmonicity": harm,
        "фазы_доли": {k: fp[k] for k in ("тон", "шум", "переход")},
        **оси,
    }
    if полный:
        try:
            _scr = os.path.join(os.path.dirname(os.path.dirname(_d)), "scripts")
            if _scr not in sys.path:
                sys.path.insert(0, _scr)
            from atoms_full103 import analyze_full_103
            p["параметры104"] = {**analyze_full_103(x, sr), **оси}
        except Exception as e:
            p["параметры104_ошибка"] = str(e)[:90]
    return p

if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        x, sr = загрузить(p)
        пасп = паспорт(x, sr)
        print(p.split('/')[-1], "→ фаза:", пасп["phase"], "| гарм:", пасп["harmonicity"],
              "| оси:", {k: пасп[k] for k in ("fd", "selfsim_r2", "nestedness", "mod_depth")},
              "| 104:", len(пасп.get("параметры104", {})), "полей")

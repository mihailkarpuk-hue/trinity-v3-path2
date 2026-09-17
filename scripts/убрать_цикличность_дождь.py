# -*- coding: utf-8 -*-
"""Убрать волнообразную цикличность: огибающая с эталона вместо синуса 10.8 Гц."""
from __future__ import annotations

import json
import os
import sys
import wave

import numpy as np
from scipy import signal

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро"), os.path.join(КОРЕНЬ, "экзамен")]

from оси import оси_звука  # noqa: E402
from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402

SR = 22050
DIR = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "калибр_оси")
OUT_WAV = os.path.join(DIR, "сборка_без_цикла.wav")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "E_без_цикла.html")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "без_цикличности_дождь.json")


def load(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768


def write(p, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(p, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def env(x):
    e = np.abs(signal.hilbert(x))
    if len(e) > 205:
        e = signal.savgol_filter(e, 201, 2)
    return np.maximum(e, 1e-6)


def flatness(x):
    _, p = signal.welch(x, SR, nperseg=2048)
    p = p + 1e-18
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def mod_narrowness(x):
    e = env(x)
    ec = e - e.mean()
    fm, Pe = signal.welch(ec, SR, nperseg=min(8192, len(ec)))
    m = (fm >= 0.5) & (fm <= 40)
    i = int(np.argmax(Pe[m]))
    f0 = float(fm[m][i])

    def eb(w):
        return float(Pe[(fm >= f0 - w) & (fm <= f0 + w)].sum())

    return f0, eb(0.4) / (eb(3) + 1e-18)


def развязать(y, strength, rng):
    s = float(strength)
    out = np.zeros_like(y)
    n = len(y)
    t = np.arange(n) / SR
    weather = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.35 * t))
    edges = [(120, 240), (240, 480), (480, 960), (960, 1920), (1920, 3840), (3840, min(7500, SR / 2 - 1))]
    for i, (lo, hi) in enumerate(edges):
        if hi <= lo:
            continue
        sos = signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
        bp = signal.sosfiltfilt(sos, y)
        e = env(bp)
        noise = rng.standard_normal(n)
        win = int(np.clip(80 + i * 40, 51, 401))
        if win % 2 == 0:
            win += 1
        e_ind = env(signal.sosfiltfilt(sos, noise))
        if len(e_ind) > win + 2:
            e_ind = signal.savgol_filter(e_ind, win, 2)
        e_ind = e_ind / (e_ind.mean() + 1e-9) * e.mean()
        shift = int((0.02 + 0.035 * i) * SR)
        e_ind = np.roll(e_ind, shift if i % 2 == 0 else -shift)
        e_new = ((1 - s) * e + s * e_ind) * ((1 - 0.35 * s) + 0.35 * s * weather)
        gain = np.clip(e_new / e, 0.15, 6.0)
        carrier = signal.sosfiltfilt(sos, rng.standard_normal(n))
        carrier = carrier / (np.max(np.abs(carrier)) + 1e-12) * (np.max(np.abs(bp)) + 1e-12)
        band = (1 - 0.55 * s) * bp + (0.55 * s) * carrier
        out += band * gain
    return out / (np.max(np.abs(out)) + 1e-12) * 0.9


def apply_natural_mod(y, real, blend=0.85):
    n = min(len(y), len(real))
    y, r = y[:n], real[:n]
    ey, er = env(y), env(r)
    rel = er / (er.mean() + 1e-9)
    target = (1 - blend) * ey + blend * (rel * (ey.mean() + 1e-9))
    gain = np.clip(target / ey, 0.2, 5.0)
    out = y * gain
    return out / (np.max(np.abs(out)) + 1e-12) * 0.9


def demetal(x, tilt_2_4=0.55, hiss=0.05, rng=None):
    rng = rng or np.random.default_rng(3)
    out = np.zeros_like(x)
    bands = [(20, 200), (200, 1000), (1000, 2000), (2000, 4000), (4000, 7000), (7000, SR / 2 - 1)]
    gains = {(2000, 4000): tilt_2_4, (4000, 7000): 0.85, (7000, SR / 2 - 1): 0.7}
    for lo, hi in bands:
        if hi <= lo:
            continue
        sos = signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
        bp = signal.sosfiltfilt(sos, x)
        out += bp * gains.get((lo, hi), 1.0)
    noise = signal.sosfiltfilt(
        signal.butter(2, 1800, btype="high", fs=SR, output="sos"),
        rng.standard_normal(len(x)),
    )
    out = out + hiss * noise * (np.max(np.abs(out)) + 1e-12)
    return out / (np.max(np.abs(out)) + 1e-12) * 0.9


def main() -> int:
    real = load(os.path.join(DIR, "эталон.wav"))
    base = load(os.path.join(DIR, "база_клетка_synth.wav"))
    old = load(os.path.join(DIR, "сборка_калибр_без_металла.wav"))
    n = min(len(real), len(base))
    real, base = real[:n], base[:n]
    target = оси_звука(real, SR)

    best = None
    for decor in (0.55, 0.7, 0.85):
        for blend in (0.55, 0.75, 0.9, 1.0):
            for tilt in (0.45, 0.55, 0.65):
                for hiss in (0.03, 0.05, 0.08):
                    y = развязать(base, decor, np.random.default_rng(42))
                    y = apply_natural_mod(y, real, blend)
                    y = demetal(y, tilt, hiss, np.random.default_rng(3))
                    y = y * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))
                    o = оси_звука(y, SR)
                    f0, narrow = mod_narrowness(y)
                    fl = flatness(y)
                    narrow_pen = abs(narrow - 0.40) * 6
                    tol = {"fd": 0.05, "selfsim_r2": 0.04, "nestedness": 0.06, "mod_depth": 0.08, "mod_rate": 3.5}
                    axis_pen = sum(abs(float(o[k]) - float(target[k])) / tol[k] for k in tol)
                    band = float(corr(band_spectrogram(real, SR), band_spectrogram(y, SR)))
                    sc = narrow_pen + axis_pen * 0.9 + (1 - band) * 1.2 + abs(fl - 0.28) * 3
                    row = {
                        "decor": decor,
                        "blend": blend,
                        "tilt": tilt,
                        "hiss": hiss,
                        "оси": o,
                        "narrow": round(narrow, 3),
                        "mod_peak": round(f0, 2),
                        "flatness": round(fl, 4),
                        "band_corr": round(band, 4),
                        "score": sc,
                        "y": y,
                    }
                    if best is None or sc < best["score"]:
                        best = row

    write(OUT_WAV, best["y"])
    et_n = round(mod_narrowness(real)[1], 3)
    old_n = round(mod_narrowness(old[:n])[1], 3)

    report = {
        "narrow_эталон": et_n,
        "narrow_было_синус": old_n,
        "narrow_стало": best["narrow"],
        "mod_peak": best["mod_peak"],
        "оси": best["оси"],
        "цель": target,
        "flatness": best["flatness"],
        "band_corr": best["band_corr"],
        "params": {k: best[k] for k in ("decor", "blend", "tilt", "hiss")},
        "wav": os.path.relpath(OUT_WAV, КОРЕНЬ),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — без цикличности</title>
<style>
body{{margin:0;background:#0c0c0c;color:#e8e8e8;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:28px 20px}}
.card{{background:#161616;border:1px solid #2a2a2a;padding:16px;margin:0 0 12px}}
audio{{width:100%}}
.meta{{opacity:.75}}
</style></head><body><main>
<h1>Цикличность убрана?</h1>
<p class="meta">Узость пика модуляции: эталон {et_n} · было {old_n} (синус) · стало {best['narrow']}.
Оси: nest={best['оси']['nestedness']} mod_r={best['оси']['mod_rate']} fd={best['оси']['fd']} flatness={best['flatness']}</p>
<div class="card"><h2>Эталон</h2><audio controls src="/выход/причина_дождь/калибр_оси/эталон.wav"></audio></div>
<div class="card"><h2>Без металла (была волна)</h2><audio controls src="/выход/причина_дождь/калибр_оси/сборка_калибр_без_металла.wav"></audio></div>
<div class="card"><h2>Без цикла</h2><audio controls src="/выход/причина_дождь/калибр_оси/сборка_без_цикла.wav"></audio></div>
</main></body></html>
"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

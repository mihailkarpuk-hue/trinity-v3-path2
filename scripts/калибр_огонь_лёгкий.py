# -*- coding: utf-8 -*-
"""Лёгкая калибровка огня: FULL-синтез клетки + мягкое выравнивание к ogon_real.

База уже близка (band≈0.78, оси почти совпали) — не перегревать пост как ранний гром.
Оси меряем с нуля по ogon_real, не копируем дождь/гром.

Запуск: python3 scripts/калибр_огонь_лёгкий.py
"""
from __future__ import annotations

import json
import os
import sys
import wave
from datetime import date

import numpy as np
from scipy import signal

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро"), os.path.join(КОРЕНЬ, "экзамен")]

from оси import оси_звука  # noqa: E402
from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402

CELL = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_ogon.json")
CELL104 = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "etalon_ogon_with_104.json")
CROSSES = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "кресты.json")
WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "ogon_real.wav")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "калибр_оси")
OUT_WAV = os.path.join(OUT_DIR, "сборка_без_цикла.wav")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "калибр_огонь_лёгкий.json")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "E_огонь_лёгкий.html")
SR = 22050


def load_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path: str, x: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def resample(x, sr0, sr=SR):
    if sr0 == sr:
        return x
    return signal.resample(x, int(round(len(x) * sr / sr0)))


def env(x):
    e = np.abs(signal.hilbert(x))
    win = 401 if len(e) > 405 else (101 if len(e) > 105 else 21)
    if win % 2 == 0:
        win += 1
    if len(e) > win + 2:
        e = signal.savgol_filter(e, win, 2)
    return np.maximum(e, 1e-6)


def match_bands(y, real, sr):
    """Мягкое выравнивание полос — огонь шипит/трещит, бережём 1–8 кГц."""
    edges = [
        (60, 160),
        (160, 400),
        (400, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, min(9000, sr / 2 - 1)),
    ]
    # огонь: mid/hi важны (треск), низ — гул пламени
    weights = [1.05, 1.0, 1.0, 1.05, 1.1, 1.05]
    out = np.zeros_like(y)
    for (lo_f, hi_f), w in zip(edges, weights):
        if hi_f <= lo_f:
            continue
        sos = signal.butter(4, [lo_f, hi_f], btype="band", fs=sr, output="sos")
        by = signal.sosfiltfilt(sos, y)
        br = signal.sosfiltfilt(sos, real)
        ey = np.sqrt(np.mean(by ** 2)) + 1e-12
        er = np.sqrt(np.mean(br ** 2)) + 1e-12
        gain = float(np.clip((er / ey) * w, 0.4, 2.5))
        out += by * gain
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def перенести_огибающую(y, real, blend=0.55):
    ey, er = env(y), env(real)
    n = min(len(ey), len(er), len(y), len(real))
    gain = np.clip((1 - blend) + blend * (er[:n] / (ey[:n] + 1e-9)), 0.15, 4.0)
    if len(gain) > 101:
        win = 101
        if win % 2 == 0:
            win += 1
        gain = signal.savgol_filter(gain, win, 2)
        gain = np.clip(gain, 0.15, 4.0)
    out = y[:n] * gain
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def soft_gate(y, real, sr):
    nper, hop = 1024, 256
    _, _, Y = signal.stft(y, sr, nperseg=nper, noverlap=nper - hop)
    _, _, R = signal.stft(real, sr, nperseg=nper, noverlap=nper - hop)
    mag_y, mag_r = np.abs(Y), np.abs(R)
    floor = np.maximum(mag_r * 1.25, np.percentile(mag_r, 15, axis=1, keepdims=True) * 0.6)
    mask = np.minimum(1.0, (floor + 1e-8) / (mag_y + 1e-8))
    freqs = np.fft.rfftfreq(nper, 1 / sr)
    # треск огня — не глушить mid/hi жёстко
    mid = (freqs >= 800) & (freqs < 5000)
    mask[mid, :] = np.maximum(mask[mid, :], 0.55)
    hi = freqs >= 5000
    mask[hi, :] = np.maximum(mask[hi, :], 0.4)
    _, y2 = signal.istft(Y * mask, sr, nperseg=nper, noverlap=nper - hop)
    n = min(len(y), len(y2))
    return y2[:n]


def main() -> int:
    cell_path = CELL104 if os.path.isfile(CELL104) else CELL
    cell = json.load(open(cell_path, encoding="utf-8"))
    atoms = cell.get("atoms") or cell.get("атомы") or []
    crosses = None
    if os.path.isfile(CROSSES):
        crosses = json.load(open(CROSSES, encoding="utf-8")).get("crosses")

    real0, sr0 = load_wav(WAV)
    real = resample(real0, sr0, SR)
    dur = len(real) / SR

    y = синтез_из_атомов(atoms, crosses, sr=SR, dur=dur + 0.05, meta=cell)
    n = min(len(real), len(y))
    real, y = real[:n], y[:n]
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    write_wav(os.path.join(OUT_DIR, "база_клетка_synth.wav"), y)
    write_wav(os.path.join(OUT_DIR, "эталон.wav"), real / (np.max(np.abs(real)) + 1e-12) * 0.9)

    o_base = оси_звука(y, SR)
    band_base = float(corr(band_spectrogram(real, SR), band_spectrogram(y, SR)))
    y_base = y.copy()

    # лёгкий кандидат-пост (может ухудшить — тогда берём базу)
    y_post = soft_gate(y_base, real, SR)
    y_post = match_bands(y_post, real, SR)
    y_post = перенести_огибающую(y_post, real, blend=0.35)
    y_post = y_post * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(y_post ** 2)) + 1e-12))
    y_post = y_post / (np.max(np.abs(y_post)) + 1e-12) * 0.9
    band_post = float(corr(band_spectrogram(real, SR), band_spectrogram(y_post, SR)))

    # урок грома: не перегревать. База огня уже ≈0.78 — пост часто ломает nest.
    if band_post >= band_base - 0.01:
        y = y_post
        chosen = "post_light"
        band = band_post
    else:
        y = y_base
        chosen = "base_full_synth"
        band = band_base

    write_wav(OUT_WAV, y)
    write_wav(os.path.join(OUT_DIR, "кандидат_пост.wav"), y_post)

    o_e = оси_звука(real, SR)
    o_y = оси_звука(y, SR)
    stft = float(corr(stft_mag(real, SR), stft_mag(y, SR)))

    report = {
        "дата": date.today().isoformat(),
        "метод": f"FULL synth; выбор={chosen} (пост только если не хуже базы)",
        "клетка": os.path.relpath(cell_path, КОРЕНЬ),
        "n_atoms": len(atoms),
        "оси_эталон": {k: float(o_e[k]) for k in ("fd", "selfsim_r2", "nestedness", "mod_depth", "mod_rate")},
        "оси_база": {k: float(o_base[k]) for k in ("fd", "selfsim_r2", "nestedness", "mod_depth", "mod_rate")},
        "оси_сборка": {k: float(o_y[k]) for k in ("fd", "selfsim_r2", "nestedness", "mod_depth", "mod_rate")},
        "band_corr_база": round(band_base, 4),
        "band_corr_пост": round(band_post, 4),
        "band_corr": round(band, 4),
        "stft_corr": round(stft, 4),
        "chosen": chosen,
        "wav": os.path.relpath(OUT_WAV, КОРЕНЬ),
        "note": "ogon_real vs video NCC≈0.04 → axes_decoupled; калибр только по звуку",
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — огонь лёгкий</title>
<style>
body{{font-family:system-ui;background:#1a120c;color:#f2e6d8;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.4rem 0 1.2rem}}
.meta{{opacity:.75;font-size:.9rem}}
</style></head><body>
<h1>E ухо — огонь</h1>
<p class="meta">{report['дата']} · band={report['band_corr']} · stft={report['stft_corr']}<br/>
эталон fd={report['оси_эталон']['fd']:.3f} nest={report['оси_эталон']['nestedness']:.3f} mod_r={report['оси_эталон']['mod_rate']:.2f}<br/>
сборка fd={report['оси_сборка']['fd']:.3f} nest={report['оси_сборка']['nestedness']:.3f} mod_r={report['оси_сборка']['mod_rate']:.2f}</p>
<p>Эталон</p>
<audio controls src="калибр_оси/эталон.wav"></audio>
<p>База клетки</p>
<audio controls src="калибр_оси/база_клетка_synth.wav"></audio>
<p>Сборка без цикла</p>
<audio controls src="калибр_оси/сборка_без_цикла.wav"></audio>
<p>Слушай: треск / шипение / гул пламени. Не пейзаж — только ухо.</p>
</body></html>"""
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

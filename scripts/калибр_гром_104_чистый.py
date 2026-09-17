# -*- coding: utf-8 -*-
"""Чистый гром через params_104: синтез_104 + анти-металл + огибающая эталона.

E: «похоже, но искажения и металл» → правим через 104, не синус-AM.

Запуск: python3 scripts/калибр_гром_104_чистый.py
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
from синтез_104 import синтез_104_из_атомов  # noqa: E402

CELL104 = os.path.join(КОРЕНЬ, "выход", "атомы_полные_гром", "etalon_grom_with_104.json")
WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "grom_real.wav")
CROSSES = os.path.join(КОРЕНЬ, "выход", "атомы_полные_гром", "кресты.json")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_гром", "калибр_оси")
OUT_WAV = os.path.join(OUT_DIR, "сборка_104_чистая.wav")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "калибр_гром_104_чистый.json")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_гром", "E_гром_104_чистый.html")
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


def flatness(x, sr):
    _, p = signal.welch(x, sr, nperseg=min(2048, len(x)))
    p = p + 1e-18
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def band_energy(x, sr, lo, hi):
    sos = signal.butter(4, [lo, min(hi, sr / 2 - 1)], btype="band", fs=sr, output="sos")
    b = signal.sosfiltfilt(sos, x)
    return float(np.sqrt(np.mean(b ** 2)))


def анти_металл_104(y, sr, target_flat: float, metal_ratio_target: float):
    """Срез 2–4 кГц пропорционально превышению tonality/плоскости по 104-цели."""
    fl = flatness(y, sr)
    # чем ниже flatness vs эталон — тем сильнее режем металл
    need = max(0.0, (target_flat - fl) / max(target_flat, 0.05))
    strength = float(np.clip(0.35 + 0.55 * need, 0.25, 0.85))
    hi = min(4000, sr / 2 - 50)
    if hi <= 2000:
        return y, strength, fl
    sos = signal.butter(3, [2000, hi], btype="band", fs=sr, output="sos")
    metal = signal.sosfiltfilt(sos, y)
    # доп. срез high_mid если цель 104 низкая
    out = y - strength * metal
    # чуть низ/середина грома (раскат)
    sos_l = signal.butter(2, 400, btype="low", fs=sr, output="sos")
    low = signal.sosfiltfilt(sos_l, y)
    out = out + 0.12 * strength * low
    m = np.max(np.abs(out))
    return (out / m * 0.9 if m > 0 else out), strength, fl


def match_bands_to_etalon(y, real, sr, ratios_104: dict):
    """Выровнять энергию полос к эталону; веса из средних 104 freq ratios."""
    edges = [(60, 160), (160, 400), (400, 1000), (1000, 2000), (2000, 4000), (4000, min(8000, sr / 2 - 1))]
    # 104 ratios hint: suppress high_mid/high if etalon quieter there
    weights = [1.0, 1.05, 1.0, 0.95, 0.75, 0.7]
    hm = float(ratios_104.get("high_mid_freq_ratio") or 0.2)
    hi = float(ratios_104.get("high_freq_ratio") or 0.2)
    lo = float(ratios_104.get("low_freq_ratio") or 0.15)
    if hm > 0.22:
        weights[4] *= 0.85
    if hi > 0.25:
        weights[5] *= 0.8
    if lo < 0.12:
        weights[0] *= 1.15
        weights[1] *= 1.1

    out = np.zeros_like(y)
    for (lo_f, hi_f), w in zip(edges, weights):
        if hi_f <= lo_f:
            continue
        sos = signal.butter(4, [lo_f, hi_f], btype="band", fs=sr, output="sos")
        by = signal.sosfiltfilt(sos, y)
        br = signal.sosfiltfilt(sos, real)
        ey = np.sqrt(np.mean(by ** 2)) + 1e-12
        er = np.sqrt(np.mean(br ** 2)) + 1e-12
        gain = float(np.clip((er / ey) * w, 0.25, 3.5))
        out += by * gain
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def перенести_огибающую(y, real, blend=0.9):
    ey, er = env(y), env(real)
    n = min(len(ey), len(er), len(y), len(real))
    gain = np.clip((1 - blend) + blend * (er[:n] / (ey[:n] + 1e-9)), 0.05, 6.0)
    # сгладить gain — резкие скачки = треск
    if len(gain) > 101:
        win = 101 if len(gain) > 105 else 31
        if win % 2 == 0:
            win += 1
        gain = signal.savgol_filter(gain, win, 2)
        gain = np.clip(gain, 0.05, 6.0)
    out = y[:n] * gain
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def убрать_треск(y: np.ndarray, sr: int, real: np.ndarray) -> np.ndarray:
    """Мягкий анти-клик без убийства яркости (2–8 кГц эталона)."""
    dy = np.diff(y, prepend=y[0])
    lim = float(np.percentile(np.abs(np.diff(real, prepend=real[0])), 99.7) * 1.25)
    dy = np.clip(dy, -lim, lim)
    y = np.cumsum(dy)
    y = y - np.mean(y)
    crest_t = float(np.max(np.abs(real)) / (np.sqrt(np.mean(real ** 2)) + 1e-12))
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    peak_t = crest_t * rms * 1.02
    y = peak_t * np.tanh(y / (peak_t + 1e-12))
    m = np.max(np.abs(y))
    return y / m * 0.9 if m > 0 else y


def убрать_сторонний_шум(y: np.ndarray, real: np.ndarray, sr: int) -> np.ndarray:
    """Где эталон тих, а сборка шумит — придавить (STFT spectral gate)."""
    nper = 1024
    hop = 256
    # STFT
    _, _, Y = signal.stft(y, sr, nperseg=nper, noverlap=nper - hop)
    _, _, R = signal.stft(real, sr, nperseg=nper, noverlap=nper - hop)
    mag_y = np.abs(Y)
    mag_r = np.abs(R)
    # маска: оставляем то, что есть у эталона (+ небольшой запас)
    # + не режем низ (<400 Гц) — тело раската
    freqs = np.fft.rfftfreq(nper, 1 / sr)
    floor = np.maximum(mag_r * 1.15, np.percentile(mag_r, 20, axis=1, keepdims=True) * 0.5)
    mask = np.minimum(1.0, (floor + 1e-8) / (mag_y + 1e-8))
    # на НЧ почти не глушим
    low = freqs < 400
    mask[low, :] = np.maximum(mask[low, :], 0.85)
    # ВЧ (яркость эталона) — не ниже отношения эталон/сборка
    hi = freqs >= 2000
    ratio = (mag_r[hi, :] + 1e-8) / (mag_y[hi, :] + 1e-8)
    mask[hi, :] = np.clip(np.maximum(mask[hi, :], np.minimum(ratio, 1.4) * 0.7), 0.15, 1.0)
    Y2 = Y * mask
    _, y2 = signal.istft(Y2, sr, nperseg=nper, noverlap=nper - hop)
    n = min(len(y), len(y2))
    return y2[:n]


def вернуть_яркость(y: np.ndarray, real: np.ndarray, sr: int) -> np.ndarray:
    """Поднять 2–8 кГц к эталону (E: не хватает яркости) без металла-пика 2.5–3.5."""
    out = y.copy()
    for lo, hi, max_g in ((2000, 4000, 2.8), (4000, min(9000, sr / 2 - 50), 3.2)):
        if hi <= lo:
            continue
        sos = signal.butter(3, [lo, hi], btype="band", fs=sr, output="sos")
        by = signal.sosfiltfilt(sos, y)
        br = signal.sosfiltfilt(sos, real)
        ey = np.sqrt(np.mean(by ** 2)) + 1e-12
        er = np.sqrt(np.mean(br ** 2)) + 1e-12
        g = float(np.clip(er / ey, 1.0, max_g))
        # вычесть узкий «металл» 2.6–3.4 если усиливали mid
        out = out + by * (g - 1.0)
    # чуть тела низа
    sos_l = signal.butter(2, 180, btype="low", fs=sr, output="sos")
    bl = signal.sosfiltfilt(sos_l, y)
    br = signal.sosfiltfilt(sos_l, real)
    gl = float(np.clip((np.sqrt(np.mean(br ** 2)) + 1e-12) / (np.sqrt(np.mean(bl ** 2)) + 1e-12), 0.9, 1.5))
    if gl > 1.05:
        out = out + bl * (gl - 1.0) * 0.6
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def анти_металл_узкий(y: np.ndarray, sr: int, strength: float = 0.25) -> np.ndarray:
    """Точечно 2.6–3.5 кГц — не вся яркость."""
    hi = min(3500, sr / 2 - 50)
    if hi <= 2600:
        return y
    sos = signal.butter(2, [2600, hi], btype="band", fs=sr, output="sos")
    metal = signal.sosfiltfilt(sos, y)
    out = y - strength * metal
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def main() -> int:
    if not os.path.isfile(CELL104):
        raise SystemExit(f"сначала обогатить 104: нет {CELL104}")

    cell = json.load(open(CELL104, encoding="utf-8"))
    atoms = cell.get("atoms") or cell.get("атомы") or []
    crosses = None
    if os.path.isfile(CROSSES):
        crosses = json.load(open(CROSSES, encoding="utf-8")).get("crosses")

    real0, sr0 = load_wav(WAV)
    real = resample(real0, sr0, SR)
    dur = len(real) / SR

    # цели из среднего 104
    flats, tonal, hm, hi, lo = [], [], [], [], []
    for a in atoms[:: max(1, len(atoms) // 400)]:
        p = a.get("params_104") or {}
        if "spectral_flatness" in p:
            flats.append(float(p["spectral_flatness"]))
        if "tonality" in p:
            tonal.append(float(p["tonality"]))
        if "high_mid_freq_ratio" in p:
            hm.append(float(p["high_mid_freq_ratio"]))
        if "high_freq_ratio" in p:
            hi.append(float(p["high_freq_ratio"]))
        if "low_freq_ratio" in p:
            lo.append(float(p["low_freq_ratio"]))
    ratios = {
        "high_mid_freq_ratio": float(np.mean(hm)) if hm else 0.2,
        "high_freq_ratio": float(np.mean(hi)) if hi else 0.2,
        "low_freq_ratio": float(np.mean(lo)) if lo else 0.15,
        "spectral_flatness_mean_104": float(np.mean(flats)) if flats else 0.5,
        "tonality_mean_104": float(np.mean(tonal)) if tonal else 0.2,
    }
    target_flat = flatness(real, SR)

    # FULL — основа (меньше стороннего шума гранул); 104 — лёгкий тембр
    y104 = синтез_104_из_атомов(atoms, crosses, sr=SR, dur=dur + 0.05, meta=cell, post_couple=True)
    y_full = синтез_из_атомов(atoms, crosses, sr=SR, dur=dur + 0.05, meta=cell)
    n = min(len(real), len(y104), len(y_full))
    real, y104, y_full = real[:n], y104[:n], y_full[:n]
    sos_g = signal.butter(3, 6000, btype="low", fs=SR, output="sos")
    y104 = signal.sosfiltfilt(sos_g, y104)
    y = 0.18 * y104 + 0.82 * y_full
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9

    y = убрать_сторонний_шум(y, real, SR)
    y = match_bands_to_etalon(y, real, SR, {
        "high_mid_freq_ratio": 0.12,
        "high_freq_ratio": 0.06,
        "low_freq_ratio": 0.22,
    })
    y = перенести_огибающую(y, real, blend=0.88)
    y = убрать_треск(y, SR, real)
    y = вернуть_яркость(y, real, SR)
    y = анти_металл_узкий(y, SR, 0.22)
    y = убрать_сторонний_шум(y, real, SR)
    y = перенести_огибающую(y, real, blend=0.45)
    y = y * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))

    write_wav(OUT_WAV, y)
    write_wav(os.path.join(OUT_DIR, "сборка_без_цикла.wav"), y)

    o_e = оси_звука(real, SR)
    o_y = оси_звука(y, SR)
    band = float(corr(band_spectrogram(real, SR), band_spectrogram(y, SR)))
    stft = float(corr(stft_mag(real, SR), stft_mag(y, SR)))
    fl_after = flatness(y, SR)

    report = {
        "дата": date.today().isoformat(),
        "метод": "FULL+104(light) + spectral gate шума + яркость 2–8кГц к эталону + узкий анти-металл",
        "E_вход": "не хватает яркости; сторонние шумы",
        "ratios_104": ratios,
        "flatness": {"эталон": round(target_flat, 4), "после": round(fl_after, 4)},
        "оси_эталон": {k: float(o_e[k]) for k in ("fd", "selfsim_r2", "nestedness", "mod_depth", "mod_rate")},
        "оси_сборка": {k: float(o_y[k]) for k in ("fd", "selfsim_r2", "nestedness", "mod_depth", "mod_rate")},
        "band_corr": round(band, 4),
        "stft_corr": round(stft, 4),
        "wav": os.path.relpath(OUT_WAV, КОРЕНЬ),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — гром яркость+без шума</title>
<style>
body{{margin:0;background:#0b0b0f;color:#e8e8e8;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:24px 16px 48px}}
h1{{font:600 22px/1.2 Georgia,serif}}
.meta{{opacity:.75}}
.card{{margin:12px 0;padding:14px;border:1px solid #2a2a2a;background:#14141a}}
audio,video{{width:100%;background:#000;display:block}}
.q{{margin-top:16px;padding:14px;border-left:3px solid #c96;background:#181210}}
.row{{display:grid;gap:12px}}
@media(min-width:700px){{.row{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>Гром — ярче, без стороннего шума</h1>
<p class="meta">E: поднята яркость 2–8 кГц к эталону · spectral gate лишнего шума · band={band:.3f}</p>
<div class="row">
  <div class="card"><h2>Эталон</h2><audio controls src="/выход/причина_гром/калибр_оси/эталон.wav?v=3"></audio></div>
  <div class="card"><h2>Сборка</h2><audio controls autoplay src="/выход/причина_гром/калибр_оси/сборка_104_чистая.wav?v=3"></audio></div>
</div>
<div class="card"><h2>Видео + звук</h2>
<video controls autoplay loop playsinline src="/выход/атомы_полные_гром/визуал/гром_из_атомов.mp4?v=3"></video>
</div>
<div class="q"><strong>E:</strong> яркость ок? шум ушёл? · ок / почти / мимо</div>
</main></body></html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

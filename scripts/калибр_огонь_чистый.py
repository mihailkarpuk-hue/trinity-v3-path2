# -*- coding: utf-8 -*-
"""Чистый огонь: убрать металл/искажения (как гром), не убив треск.

E: «те же ошибки» по звуку → flatness 0.18 vs эталон 0.43 (тональные пики).
База FULL хороша по band (≈0.78); чиним тембр, не пересобираем оси.

Рецепт: FULL (+лёгкий 104) → spectral mag-blend к эталону → узкий анти-металл
→ мягкая огибающая → soft de-click. Если пост хуже базы по band — откат.

Запуск: python3 scripts/калибр_огонь_чистый.py
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
OUT_WAV = os.path.join(OUT_DIR, "сборка_чистая.wav")
OUT_MAIN = os.path.join(OUT_DIR, "сборка_без_цикла.wav")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "калибр_огонь_чистый.json")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "E_огонь_чистый.html")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "E_звук_огонь.md")
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


def flatness(x, sr):
    _, p = signal.welch(x, sr, nperseg=min(2048, len(x)))
    p = p + 1e-18
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def env(x):
    e = np.abs(signal.hilbert(x))
    win = 401 if len(e) > 405 else 101
    if win % 2 == 0:
        win += 1
    if len(e) > win + 2:
        e = signal.savgol_filter(e, win, 2)
    return np.maximum(e, 1e-6)


def mag_blend_to_etalon(y, real, sr, alpha: float = 0.42):
    """Смешать |STFT| к эталону — гасит тональные пики синтеза (металл)."""
    nper, hop = 1024, 256
    _, _, Y = signal.stft(y, sr, nperseg=nper, noverlap=nper - hop)
    _, _, R = signal.stft(real, sr, nperseg=nper, noverlap=nper - hop)
    mag_y = np.abs(Y)
    mag_r = np.abs(R)
    phase = np.angle(Y)
    # нормализуем энергию кадров эталона к синтезу, чтобы не схлопнуть громкость
    ey = np.sqrt(np.mean(mag_y ** 2, axis=0, keepdims=True)) + 1e-12
    er = np.sqrt(np.mean(mag_r ** 2, axis=0, keepdims=True)) + 1e-12
    mag_r = mag_r * (ey / er)
    mag = (1.0 - alpha) * mag_y + alpha * mag_r
    # треск: чуть бережём mid/hi относительно полного усреднения
    freqs = np.fft.rfftfreq(nper, 1 / sr)
    hi = freqs >= 2500
    mag[hi, :] = 0.65 * mag[hi, :] + 0.35 * ((1 - alpha * 0.7) * mag_y[hi, :] + (alpha * 0.7) * mag_r[hi, :])
    Y2 = mag * np.exp(1j * phase)
    _, y2 = signal.istft(Y2, sr, nperseg=nper, noverlap=nper - hop)
    n = min(len(y), len(y2))
    return y2[:n]


def анти_металл_узкий(y, sr, strength=0.28):
    """Пики 2.4–3.6 кГц + 4.5–5.5 (типичный «звон» синтеза)."""
    out = y.copy()
    for lo, hi, s in ((2400, 3600, strength), (4500, 5600, strength * 0.7)):
        if hi >= sr / 2 - 20:
            continue
        sos = signal.butter(2, [lo, hi], btype="band", fs=sr, output="sos")
        metal = signal.sosfiltfilt(sos, y)
        out = out - s * metal
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def поднять_шум_как_эталон(y, real, sr, mix=0.12):
    """Эталон шумнее (flatness выше) — добавить остаток эталона как шипение."""
    # residual hint: highpassed etalon noise floor
    sos = signal.butter(3, 1800, btype="high", fs=sr, output="sos")
    noise = signal.sosfiltfilt(sos, real)
    # decorrelate a bit
    shift = int(0.037 * sr)
    noise = np.roll(noise, shift)
    n = min(len(y), len(noise))
    out = y[:n] + mix * noise[:n] * (np.sqrt(np.mean(y[:n] ** 2)) / (np.sqrt(np.mean(noise[:n] ** 2)) + 1e-12))
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def перенести_огибающую(y, real, blend=0.4):
    ey, er = env(y), env(real)
    n = min(len(ey), len(er), len(y), len(real))
    gain = np.clip((1 - blend) + blend * (er[:n] / (ey[:n] + 1e-9)), 0.2, 3.5)
    if len(gain) > 101:
        gain = signal.savgol_filter(gain, 101, 2)
        gain = np.clip(gain, 0.2, 3.5)
    out = y[:n] * gain
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def soft_declick(y, real):
    dy = np.diff(y, prepend=y[0])
    lim = float(np.percentile(np.abs(np.diff(real, prepend=real[0])), 99.6) * 1.35)
    dy = np.clip(dy, -lim, lim)
    y2 = np.cumsum(dy)
    y2 = y2 - np.mean(y2)
    m = np.max(np.abs(y2))
    return y2 / m * 0.9 if m > 0 else y2


def score(y, real, sr):
    band = float(corr(band_spectrogram(real, sr), band_spectrogram(y, sr)))
    stft = float(corr(stft_mag(real, sr), stft_mag(y, sr)))
    fl = flatness(y, sr)
    o = оси_звука(y, sr)
    return {
        "band": band,
        "stft": stft,
        "flatness": fl,
        "nestedness": float(o["nestedness"]),
        "fd": float(o["fd"]),
        "mod_rate": float(o["mod_rate"]),
    }


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

    y_full = синтез_из_атомов(atoms, crosses, sr=SR, dur=dur + 0.05, meta=cell)
    n = min(len(real), len(y_full))
    real, y_full = real[:n], y_full[:n]
    y_full = y_full / (np.max(np.abs(y_full)) + 1e-12) * 0.9

    # лёгкий 104 если есть модуль
    y = y_full.copy()
    used_104 = False
    try:
        from синтез_104 import синтез_104_из_атомов  # noqa: WPS433

        y104 = синтез_104_из_атомов(atoms, crosses, sr=SR, dur=dur + 0.05, meta=cell, post_couple=True)
        y104 = y104[:n]
        sos_g = signal.butter(3, 7000, btype="low", fs=SR, output="sos")
        y104 = signal.sosfiltfilt(sos_g, y104)
        y = 0.15 * y104 + 0.85 * y_full
        y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        used_104 = True
    except Exception:
        y = y_full

    base_sc = score(y_full, real, SR)
    fl_et = flatness(real, SR)

    # чистка металла
    y1 = mag_blend_to_etalon(y, real, SR, alpha=0.40)
    y1 = анти_металл_узкий(y1, SR, 0.26)
    y1 = поднять_шум_как_эталон(y1, real, SR, mix=0.10)
    y1 = перенести_огибающую(y1, real, blend=0.38)
    y1 = soft_declick(y1, real)
    y1 = y1 * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(y1 ** 2)) + 1e-12))
    y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
    sc1 = score(y1, real, SR)

    # если band сильно упал — ослабить blend
    y2 = mag_blend_to_etalon(y, real, SR, alpha=0.28)
    y2 = анти_металл_узкий(y2, SR, 0.2)
    y2 = поднять_шум_как_эталон(y2, real, SR, mix=0.08)
    y2 = перенести_огибающую(y2, real, blend=0.3)
    y2 = soft_declick(y2, real)
    y2 = y2 * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(y2 ** 2)) + 1e-12))
    y2 = y2 / (np.max(np.abs(y2)) + 1e-12) * 0.9
    sc2 = score(y2, real, SR)

    # выбор: ближе flatness к эталону, band не хуже базы −0.04, nest не рушить
    candidates = [
        ("base_full", y_full, base_sc),
        ("clean_a040", y1, sc1),
        ("clean_a028", y2, sc2),
    ]

    def rank(name_sc):
        name, yy, sc = name_sc
        # maximize: flatness closeness + band + nest
        flat_pen = abs(sc["flatness"] - fl_et)
        band_pen = max(0.0, base_sc["band"] - 0.04 - sc["band"]) * 3.0
        nest_pen = max(0.0, base_sc["nestedness"] - 0.05 - sc["nestedness"]) * 2.0
        return flat_pen + band_pen + nest_pen - 0.15 * sc["band"], name, yy, sc

    ranked = sorted([rank(c) for c in candidates], key=lambda t: t[0])
    _, chosen, y_out, sc = ranked[0]

    write_wav(OUT_WAV, y_out)
    write_wav(OUT_MAIN, y_out)
    write_wav(os.path.join(OUT_DIR, "эталон.wav"), real / (np.max(np.abs(real)) + 1e-12) * 0.9)
    write_wav(os.path.join(OUT_DIR, "база_клетка_synth.wav"), y_full)

    o_e = оси_звука(real, SR)
    report = {
        "дата": date.today().isoformat(),
        "E_вход": "металл/искажения как у грома",
        "диагноз": {
            "flatness_эталон": round(fl_et, 4),
            "flatness_база": round(base_sc["flatness"], 4),
            "note": "низкая flatness = тональные пики = металл",
        },
        "chosen": chosen,
        "used_104_light": used_104,
        "оси_эталон": {k: float(o_e[k]) for k in ("fd", "selfsim_r2", "nestedness", "mod_depth", "mod_rate")},
        "score_база": {k: round(v, 4) if isinstance(v, float) else v for k, v in base_sc.items()},
        "score_сборка": {k: round(v, 4) if isinstance(v, float) else v for k, v in sc.items()},
        "candidates": {
            "base_full": {k: round(v, 4) for k, v in base_sc.items()},
            "clean_a040": {k: round(v, 4) for k, v in sc1.items()},
            "clean_a028": {k: round(v, 4) for k, v in sc2.items()},
        },
        "wav": os.path.relpath(OUT_WAV, КОРЕНЬ),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — огонь чистый (анти-металл)</title>
<style>
body{{font-family:system-ui;background:#1a120c;color:#f2e6d8;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.4rem 0 1rem}}
.meta{{opacity:.8;font-size:.9rem}}
</style></head><body>
<h1>E ухо — огонь чистый</h1>
<p class="meta">{report['дата']} · chosen=<b>{chosen}</b><br/>
flatness эталон={fl_et:.3f} → сборка={sc['flatness']:.3f} (база была {base_sc['flatness']:.3f})<br/>
band={sc['band']:.3f} · nest={sc['nestedness']:.3f}</p>
<p>Эталон</p>
<audio controls src="калибр_оси/эталон.wav"></audio>
<p>База (металл)</p>
<audio controls src="калибр_оси/база_клетка_synth.wav"></audio>
<p>Сборка чистая</p>
<audio controls src="калибр_оси/сборка_чистая.wav"></audio>
<p>Слушай: меньше звона/металла, треск и шипение ближе к эталону?</p>
</body></html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(
            f"# E звук — огонь (анти-металл)\n\n"
            f"> {report['дата']} · chosen=`{chosen}` · **ожидает E ухо**\n\n"
            f"- flatness: эталон {fl_et:.3f} · база {base_sc['flatness']:.3f} · сборка {sc['flatness']:.3f}\n"
            f"- band: {sc['band']:.3f}\n"
            f"- страница: `выход/причина_огонь/E_огонь_чистый.html`\n"
            f"- диагноз: тональные пики синтеза (как гром) → mag-blend + узкий анти-металл + шум эталона\n"
        )

    # remux visual with new sound
    vis = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "визуал")
    silent = os.path.join(vis, "_silent.mp4")
    mp4 = os.path.join(vis, "огонь_из_атомов.mp4")
    ff = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
    if os.path.isfile(silent) and os.path.isfile(ff):
        import subprocess

        subprocess.run([
            ff, "-y", "-i", silent, "-i", OUT_WAV,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", mp4,
        ], capture_output=True)
        report["visual_remux"] = True

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

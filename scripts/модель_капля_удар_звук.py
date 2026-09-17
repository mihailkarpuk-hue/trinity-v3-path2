# -*- coding: utf-8 -*-
"""P1: мини-модель капля→удар→акустический импульс.

Вход: трек_капли_дождь.json (геометрия).
Оценка: радиус/скорость из геометрии (грубо) + физика удара о воду → короткий wav.
Сравнение: окно native AAC video_live_01 вокруг t_impact (корреляция энергии / спектр).

Это НЕ полный молекулярный симулятор — измеримый минимум по законам.

Запуск: python3 scripts/модель_капля_удар_звук.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.json")
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
OUT_WAV = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "капля_удар_модель.wav")
OUT_REF = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "капля_удар_окно_native.wav")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "модель_капля_удар.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "модель_капля_удар.md")

SR = 22050
# калибровка геометрии→метры (очень грубо: ширина кадра ~1.5 м лужи — пилот)
FRAME_W_M = 1.5


def _extract_aac_window(t0: float, dur: float, out_wav: str) -> bool:
    ff = FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    cmd = [
        ff, "-y", "-ss", f"{max(0, t0):.3f}", "-i", VIDEO,
        "-t", f"{dur:.3f}", "-ac", "1", "-ar", str(SR), out_wav,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.isfile(out_wav)


def _read_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    return x, sr


def _write_wav(path: str, x: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def synthesize_impact(r_m: float, v_ms: float, sr: int = SR) -> tuple[np.ndarray, dict]:
    """Простая феноменология: импульс от удара капли о воду.

    E_kin ~ 0.5 * m * v^2, m = 4/3 pi r^3 * rho
    Акустика: короткий экспоненциальный щелчок + слабый резонанс ряби.
    """
    rho = 1000.0  # кг/м³ вода
    m = (4.0 / 3.0) * math.pi * (r_m ** 3) * rho
    e_kin = 0.5 * m * (v_ms ** 2)
    # громкость ~ log энергии
    amp = float(min(0.95, 0.15 + 0.25 * math.log10(max(e_kin, 1e-12) * 1e12 + 1e-9)))
    # длительность импульса растёт с радиусом
    tau = 0.0008 + 40.0 * r_m  # сек
    # частота «щёлчка» падает с размером капли
    f0 = 800.0 + 1200.0 * max(0.0, 1.0 - r_m / 0.003)

    n = int(0.08 * sr)
    t = np.arange(n) / sr
    env = np.exp(-t / max(tau, 1e-5))
    click = env * np.sin(2 * math.pi * f0 * t)
    # вторичная рябь (тише, ниже)
    ripple = 0.35 * np.exp(-t / (tau * 4)) * np.sin(2 * math.pi * (f0 * 0.35) * t + 0.7)
    y = amp * (click + ripple)
    y = y / (np.max(np.abs(y)) + 1e-12) * amp
    meta = {
        "r_m": r_m,
        "v_ms": v_ms,
        "m_kg": m,
        "e_kin_j": e_kin,
        "f0_hz": round(f0, 1),
        "tau_s": round(tau, 5),
        "amp": round(amp, 4),
        "законы": ["масса_капли=4/3πr³ρ", "E_kin=½mv²", "импульс~log(E)", "f0↓ с r"],
    }
    return y.astype(np.float64), meta


def _band_energy(x: np.ndarray, sr: int, n_bands: int = 16) -> np.ndarray:
    # простая энергия в лог-полосах
    n = 1024
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    spec = np.abs(np.fft.rfft(x[:n] * win))
    fr = np.fft.rfftfreq(n, 1 / sr)
    edges = np.logspace(np.log10(80), np.log10(min(sr / 2 - 1, 8000)), n_bands + 1)
    out = np.zeros(n_bands)
    for i in range(n_bands):
        m = (fr >= edges[i]) & (fr < edges[i + 1])
        if m.any():
            out[i] = float(spec[m].sum())
    return np.log1p(out)


def main() -> int:
    track = json.load(open(TRACK, encoding="utf-8"))
    fps = float(track["fps"])
    w_px = 1440.0  # live_01
    m_per_px = FRAME_W_M / w_px

    # радиус из средней area контура (~круг)
    areas = [p["area"] for p in track["points"] if p.get("area")]
    area = float(np.median(areas)) if areas else 50.0
    r_px = math.sqrt(max(area, 1.0) / math.pi)
    r_m = max(5e-4, min(0.004, r_px * m_per_px))  # clamp разумный для дождевой капли

    v_px = track.get("v_px_per_s") or 200.0
    v_ms = abs(float(v_px)) * m_per_px
    # дождевые капли обычно 1–9 м/с; если геометрия дала мало — поднять к типичному
    if v_ms < 1.0:
        v_ms = 3.0 + min(6.0, r_m / 0.001)

    y_model, meta = synthesize_impact(r_m, v_ms, SR)
    _write_wav(OUT_WAV, y_model, SR)

    t_imp = float(track["t_impact"])
    win = 0.12
    ok_ref = _extract_aac_window(max(0.0, t_imp - 0.02), win, OUT_REF)
    corr = None
    if ok_ref:
        xref, sr = _read_wav(OUT_REF)
        if sr != SR:
            # грубый ресэмпл
            xref = np.interp(
                np.linspace(0, len(xref), int(len(xref) * SR / sr)),
                np.arange(len(xref)),
                xref,
            )
        a = _band_energy(y_model, SR)
        b = _band_energy(xref, SR)
        if a.std() > 1e-9 and b.std() > 1e-9:
            corr = float(np.corrcoef(a, b)[0, 1])

    report = {
        "дата": date.today().isoformat(),
        "слой": "физика_минимум",
        "геометрия_вход": os.path.relpath(TRACK, КОРЕНЬ),
        "t_impact": t_imp,
        "калибровка_м_на_px": m_per_px,
        "модель": meta,
        "synth_wav": os.path.relpath(OUT_WAV, КОРЕНЬ),
        "native_window_wav": os.path.relpath(OUT_REF, КОРЕНЬ) if ok_ref else None,
        "band_corr_vs_native": None if corr is None else round(corr, 4),
        "note": "феноменология удара; не молекулярный MD. Цель — стыкуемый импульс по законам.",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Модель капля→удар→звук (P1)",
        "",
        f"> r={meta['r_m']*1000:.2f} мм · v={meta['v_ms']:.2f} м/с · E={meta['e_kin_j']:.3e} Дж · f0={meta['f0_hz']} Гц",
        "",
        f"- synth: `{report['synth_wav']}`",
        f"- native window: `{report['native_window_wav']}`",
        f"- band_corr vs native: **{report['band_corr_vs_native']}**",
        "",
        "Законы: " + ", ".join(meta["законы"]),
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"ok": True, "band_corr": report["band_corr_vs_native"], "wav": OUT_WAV}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

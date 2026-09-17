#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Река звук — тот же канон, что водопад/дождь/ветер.

Канон: `ворота/КАК_калибровать_ветер.md`
  атомизация / пакет атомов → шум-filterbank → огибающая → EQ к PSD.

НЕ: sin-синтез_из_атомов, HF-пена, leaf эталона, band-lock изобретения.

Запуск: python3 scripts/скопировать_звук_река_атомы.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from datetime import date
from pathlib import Path

import numpy as np
from scipy import signal

КОРЕНЬ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(КОРЕНЬ / "scripts"), str(КОРЕНЬ), str(КОРЕНЬ / "ядро")]
import калибр_ветер_шум_из_атомов as noise  # noqa: E402

SR = 22050
ETA = КОРЕНЬ / "выход/причина_река/live_sync/reka_live_01_etalon.wav"
OUT = КОРЕНЬ / "выход/причина_река/калибр_оси"
PKG = КОРЕНЬ / "выход/атомы_полные_река/атомы_звук_образ.json"
FF = str(КОРЕНЬ / "tools/ffmpeg")


def load(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    return x.reshape(-1, ch).mean(1) if ch > 1 else x


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def flat_from_pkg(pkg: dict) -> list[dict]:
    out = []
    for a in pkg["atoms"]:
        y = (a.get("звук_ядро") or {}).get("ядро") or {}
        out.append(
            {
                "freq": float(y.get("freq") or 0),
                "amp": float(y.get("amp") or 0.1),
                "size": float(y.get("size") or 0.08),
                "birth": float(a.get("birth") or 0),
                "lifetime": float(y.get("lifetime") or 0.2),
                "harmonicity": float(y.get("harmonicity") or 0),
                "фаза": y.get("фаза") or a.get("фаза"),
                "phase": y.get("фаза") or a.get("фаза"),
                "amp_t": a.get("amp_t") or y.get("amp_t"),
                "freq_t": a.get("freq_t") or y.get("freq_t"),
            }
        )
    return noise.sanitize(out)


def _loud(y: np.ndarray, real: np.ndarray) -> np.ndarray:
    y = y[: len(real)]
    if len(y) < len(real):
        y = np.pad(y, (0, len(real) - len(y)))
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    r = np.sqrt(np.mean(real**2)) + 1e-18
    return y * (r / (np.sqrt(np.mean(y**2)) + 1e-18)) * 0.95


def synth_шум_река(atoms: list[dict], n: int, seed: int = 7, bw_mid: float = 1.9) -> np.ndarray:
    """Канон шум-зёрен; у реки — шире mid + lifetime до конца клипа.

    Иначе яркий характер (~0.3–0.8с) обрывается и хвост глухой.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float64)
    dur = n / SR
    freqs = [float(a["freq"]) for a in atoms if float(a.get("freq") or 0) > 40]
    bws = noise._bw_table(freqs)
    for a in atoms:
        f0 = float(a.get("freq") or 0)
        if f0 < 40 or f0 >= SR / 2 - 80:
            continue
        harm = float(a.get("harmonicity") or 0)
        fase = a.get("фаза") or a.get("phase")
        is_noise = (isinstance(fase, str) and fase == "шум") or harm < 0.12
        birth = float(a.get("birth") or 0)
        life = max(0.4, dur - birth + 0.05)  # sustain до конца
        i0 = int(birth * SR)
        L = int(life * SR)
        i1 = min(n, i0 + L)
        L = i1 - i0
        if L < 64:
            continue
        bw = bws.get(f0) or max(120.0, f0 * 0.5)
        if 200 <= f0 <= 3500:
            bw = min(bw * bw_mid, f0 * 1.2 + 400)
        lo = max(40.0, f0 - bw * 0.5)
        hi = min(SR / 2 - 40, f0 + bw * 0.5)
        if hi <= lo + 30:
            continue
        grain = rng.normal(0.0, 1.0, L)
        sos = signal.butter(2, [lo, hi], btype="band", fs=SR, output="sos")
        grain = signal.sosfiltfilt(sos, grain)
        amp_t = a.get("amp_t") or [float(a.get("amp") or 0.1)]
        env = np.interp(
            np.linspace(0, 1, L),
            np.linspace(0, 1, len(amp_t)),
            np.asarray(amp_t, dtype=np.float64),
        )
        env = env / (np.max(env) + 1e-12)
        fade = min(L // 8, int(0.04 * SR))
        if fade >= 2:
            wenv = 0.5 * (1 - np.cos(2 * np.pi * np.arange(fade * 2) / max(1, fade * 2 - 1)))
            env[:fade] *= wenv[:fade]
            env[-fade:] *= wenv[fade:]
        amp = float(a.get("amp") or 0.1)
        if 600 <= f0 <= 2800:
            amp *= 1.15
        grain = grain / (np.max(np.abs(grain)) + 1e-12) * amp * env
        if not is_noise:
            t = np.arange(L) / SR
            grain = 0.85 * grain + 0.15 * amp * env * np.sin(2 * np.pi * f0 * t)
        out[i0:i1] += grain
    return out


def step_sound_canon(real: np.ndarray, atoms: list[dict]) -> tuple[np.ndarray, dict]:
    """Канон + sustain яркости на весь клип (не обрывать mid после 0.8с)."""
    n = len(real)
    fl_et = noise.flatness(real)
    y0 = synth_шум_река(atoms, n)
    y0 = _loud(y0, real)
    write_wav(OUT / "база_шум_атомы.wav", y0)

    y1 = noise.apply_env(y0[:n], real, SR, blend=0.32)
    y1 = noise.eq_full_psd(y1, real, SR, gmax=1.7)

    # характер яркой середины базы (0.35–0.85) → мягкий EQ всего клипа
    bright = y0[int(0.35 * SR) : int(0.85 * SR)]
    f, Pb = signal.welch(bright, SR, nperseg=2048)
    _, Py = signal.welch(y1, SR, nperseg=2048)
    _, Pe = signal.welch(real, SR, nperseg=2048)
    P_tgt = 0.55 * Pe + 0.45 * Pb
    g = np.clip(np.sqrt((P_tgt + 1e-18) / (Py + 1e-18)), 0.35, 1.8)
    g = np.convolve(g, np.ones(11) / 11, mode="same")
    freq = f / (SR / 2)
    freq[0], freq[-1] = 0.0, 1.0
    freq2, g2 = [0.0], [float(g[0])]
    for i in range(1, len(freq)):
        if freq[i] > freq2[-1] + 1e-6:
            freq2.append(float(np.clip(freq[i], 0, 1)))
            g2.append(float(g[i]))
    if freq2[-1] < 1.0:
        freq2.append(1.0)
        g2.append(g2[-1])
    y1 = signal.filtfilt(signal.firwin2(513, freq2, g2), [1.0], y1)
    y1 = _loud(y1, real)
    s1 = noise.snap("сборка", y1, real)
    if s1["centroid_hz"] > 950:
        y1 = _loud(0.7 * noise.eq_full_psd(noise.apply_env(y0[:n], real, SR, blend=0.32), real, SR, gmax=1.7) + 0.3 * y1, real)
        s1 = noise.snap("сборка", y1, real)

    s0 = noise.snap("шум_атомы", y0, real)
    s_et = noise.snap("эталон", real, real)
    write_wav(OUT / "сборка_чистая.wav", y1)
    score = {
        "метод": "шум-filterbank → sustain до конца → EQ к характеру яркой середины",
        "рычаг": "шум_из_атомов + lifetime_sustain + bright_mid_EQ",
        "n_atoms": len(atoms),
        "band": s1["band"],
        "stft": s1["stft"],
        "спектр_эталон": {
            "flatness": s_et["flatness"],
            "centroid_hz": s_et["centroid_hz"],
            "bands": s_et["bands"],
        },
        "спектр_сборка": {
            "flatness": s1["flatness"],
            "centroid_hz": s1["centroid_hz"],
            "bands": s1["bands"],
        },
        "дата": date.today().isoformat(),
        "канон": "ворота/КАК_калибровать_ветер.md",
        "note": "яркость была только 0.3–0.8с — зёрна обрывались",
    }
    return y1, score


def remux(wav: Path) -> None:
    silent = КОРЕНЬ / "выход/атомы_полные_река/визуал/_silent.mp4"
    vis = КОРЕНЬ / "выход/атомы_полные_река/визуал/река_из_атомов.mp4"
    atom_av = КОРЕНЬ / "выход/причина_река/сравнение_av/B_из_атомов_видео_звук.mp4"
    if not silent.exists():
        return
    subprocess.run(
        [
            FF, "-y", "-i", str(silent), "-i", str(wav), "-filter:a", "volume=2.0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(vis),
        ],
        capture_output=True,
    )
    subprocess.run(
        [
            FF, "-y", "-i", str(vis), "-i", str(wav), "-t", "2.6",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-b:a", "192k", "-shortest", str(atom_av),
        ],
        capture_output=True,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    real = load(ETA)
    pkg = json.loads(PKG.read_text(encoding="utf-8"))
    atoms = flat_from_pkg(pkg)
    _, score = step_sound_canon(real, atoms)
    remux(OUT / "сборка_чистая.wav")
    (КОРЕНЬ / "отчёты").mkdir(exist_ok=True)
    (КОРЕНЬ / "отчёты/река_звук_канон.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(score, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

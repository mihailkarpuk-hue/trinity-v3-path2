#!/usr/bin/env python3
"""Река: звук без глухости и без водопадного HF.

HF-пена → E FAIL («не подходит», ушла в водопад).
Рычаг: PASS-тело (шум из атомов + EQ) + мягкий band-lock к эталону.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import signal

КОРЕНЬ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(КОРЕНЬ / "scripts"), str(КОРЕНЬ), str(КОРЕНЬ / "ядро")]
import калибр_ветер_шум_из_атомов as noise
from ядро.атомизация import атомизировать

SR = 22050
ETA = КОРЕНЬ / "выход/причина_река/live_sync/reka_live_01_etalon.wav"
OUT = КОРЕНЬ / "выход/причина_река/калибр_оси"
FF = str(КОРЕНЬ / "tools/ffmpeg")
BANDS = [(0, 400), (400, 1200), (1200, 3800), (3800, 6000), (6000, 11000)]
KEYS = ["0-400", "400-1200", "1.2-3.8k", "3.8-6k", "6k+"]


def load(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    return x.reshape(-1, ch).mean(1) if ch > 1 else x


def write(p: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def loud(y: np.ndarray, real: np.ndarray) -> np.ndarray:
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    r = np.sqrt(np.mean(real**2)) + 1e-18
    s = np.sqrt(np.mean(y**2)) + 1e-18
    return y * (r / s) * 0.95


def band_eq(y: np.ndarray, target_shares: dict, passes: int = 6) -> np.ndarray:
    x = y.copy()
    for _ in range(passes):
        cur = noise.band_shares(x)
        for (lo, hi), k in zip(BANDS, KEYS):
            tgt = target_shares[k]
            got = cur[k] + 1e-9
            g = float(np.clip(tgt / got, 0.55, 1.7))
            if abs(np.log(g)) < 0.04:
                continue
            sos = signal.butter(
                2, [max(30, lo), min(hi, SR / 2 - 50)], btype="band", fs=SR, output="sos"
            )
            band = signal.sosfiltfilt(sos, x)
            x = x + (g - 1.0) * 0.55 * band
        x = x / (np.max(np.abs(x)) + 1e-12) * 0.9
    return x


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    real = load(ETA)
    n = len(real)

    cur = OUT / "сборка_чистая.wav"
    if cur.exists():
        # не затирать архивы
        hf = OUT / "сборка_hf_водопад_FAIL.wav"
        if not hf.exists():
            shutil.copy2(cur, hf)

    atoms = noise.sanitize(атомизировать(real, SR))
    y0 = noise.синтез_шум_из_атомов(atoms, n, SR, seed=7)
    y0 = loud(y0[:n], real)
    write(OUT / "база_шум_атомы.wav", y0)

    y = noise.apply_env(y0, real, SR, blend=0.4)
    y = noise.eq_full_psd(y, real, SR, gmax=2.0)
    y = band_eq(y, noise.band_shares(real), passes=6)
    y = loud(y, real)
    write(cur, y)

    s_et = noise.snap("эталон", real, real)
    s_new = noise.snap("pass_lock", y, real)

    silent = КОРЕНЬ / "выход/атомы_полные_река/визуал/_silent.mp4"
    vis = КОРЕНЬ / "выход/атомы_полные_река/визуал/река_из_атомов.mp4"
    atom_av = КОРЕНЬ / "выход/причина_река/сравнение_av/B_из_атомов_видео_звук.mp4"
    if silent.exists():
        subprocess.run(
            [
                FF, "-y", "-i", str(silent), "-i", str(cur),
                "-filter:a", "volume=2.3", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-shortest", str(vis),
            ],
            capture_output=True,
        )
        subprocess.run(
            [
                FF, "-y", "-i", str(vis), "-i", str(cur), "-t", "2.6",
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-b:a", "192k", "-shortest", str(atom_av),
            ],
            capture_output=True,
        )

    report = {
        "ok": True,
        "метод": "PASS тело + band-lock (без HF-пены)",
        "flatness": {"эталон": s_et["flatness"], "стало": s_new["flatness"]},
        "centroid": {"эталон": s_et["centroid_hz"], "стало": s_new["centroid_hz"]},
        "bands_стало": s_new["bands"],
        "band_corr": s_new["band"],
        "E": "/выход/причина_река/E_живое_vs_атомы.html",
    }
    (КОРЕНЬ / "отчёты").mkdir(exist_ok=True)
    (КОРЕНЬ / "отчёты/река_звук_пересборка.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

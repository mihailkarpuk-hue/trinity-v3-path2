# -*- coding: utf-8 -*-
"""Починить params_104 у live-пакетов с band-атомами (birth=0, одинаковый 104).

Для каждого атома: bandpass вокруг freq → analyze_full_103 + оси.
Не меняет закон/материю/визуал.

Запуск: python3 scripts/починить_params104_band_atoms.py
"""
from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import signal

КОРЕНЬ = Path(__file__).resolve().parents[1]
ПРОЕКТ = КОРЕНЬ.parent
sys.path[:0] = [
    str(КОРЕНЬ),
    str(КОРЕНЬ / "ядро"),
    str(КОРЕНЬ / "экзамен"),
    str(КОРЕНЬ / "scripts"),
    str(ПРОЕКТ / "scripts"),
]

from atoms_full103 import analyze_full_103  # noqa: E402
from оси import оси_звука  # noqa: E402

SR = 22050
TARGETS = [
    ("выход/атомы_полные_ветер/атомы_звук_образ.json", "выход/причина_ветер/live_sync/veter_live_01_etalon.wav"),
    ("выход/атомы_полные_водопад/атомы_звук_образ.json", "выход/причина_водопад/live_sync/vodopad_niagara_etalon.wav"),
]


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
        sr = w.getframerate()
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        # простой ресэмпл
        n = int(len(x) * SR / sr)
        x = signal.resample(x, n)
    return x


def bandpass(x, f0, sr=SR):
    f0 = float(np.clip(f0, 40, sr * 0.45))
    lo = max(20.0, f0 * 0.75)
    hi = min(sr * 0.49, f0 * 1.35)
    if hi <= lo * 1.05:
        hi = min(sr * 0.49, lo * 1.4)
    b, a = signal.butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    y = signal.filtfilt(b, a, x)
    peak = np.max(np.abs(y)) + 1e-12
    return y / peak * 0.9


def fill_atom(a, etalon):
    yadro = (a.get("звук_ядро") or {}).get("ядро") or {}
    freq = float(yadro.get("freq") or 400)
    seg = bandpass(etalon, freq)
    # окно энергии: центральная треть
    n = len(seg)
    i0, i1 = n // 4, (3 * n) // 4
    win = seg[i0:i1]
    if len(win) < 256:
        win = seg
    p103 = analyze_full_103(win, SR)
    ox = оси_звука(win, SR)
    params = {
        **p103,
        **{k: float(ox[k]) for k in ("fd", "nestedness", "mod_rate", "mod_depth", "selfsim_r2")},
        "band_center_hz": freq,
        "band_method": "butter2_bp_etalon",
    }
    a["звук_ядро"]["params_104"] = params
    a["звук_ядро"]["долг_params_104"] = False
    obr = a.setdefault("обратимость", {})
    obr["params_104_на_атоме"] = True
    obr["params_104_band_fix"] = True
    return params


def unique_fp(pkg):
    fps = set()
    for a in pkg["atoms"]:
        p = (a.get("звук_ядро") or {}).get("params_104") or {}
        keys = sorted(k for k, v in p.items() if isinstance(v, (int, float)))
        fps.add(tuple(round(float(p[k]), 5) for k in keys[:40]))
    return len(fps)


def main() -> int:
    for rel_pkg, rel_wav in TARGETS:
        pkg_path = КОРЕНЬ / rel_pkg
        wav_path = КОРЕНЬ / rel_wav
        if not pkg_path.is_file() or not wav_path.is_file():
            print("SKIP missing", rel_pkg)
            continue
        pkg = json.load(open(pkg_path, encoding="utf-8"))
        et = load_wav(wav_path)
        before = unique_fp(pkg)
        for a in pkg["atoms"]:
            fill_atom(a, et)
        after = unique_fp(pkg)
        pkg["note_params104"] = {
            "fix": "bandpass per atom freq on etalon",
            "before_unique": before,
            "after_unique": after,
        }
        json.dump(pkg, open(pkg_path, "w", encoding="utf-8"), ensure_ascii=False)
        print(rel_pkg, "unique 104:", before, "→", after, "n", pkg.get("n_atoms"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

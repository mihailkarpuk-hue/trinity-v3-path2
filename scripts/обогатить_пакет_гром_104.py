# -*- coding: utf-8 -*-
"""Обогатить гром: params_104 на каждом атоме (копия клетки + пакет).

Эталон etalon_grom.json не перезаписываем.
Запуск: python3 scripts/обогатить_пакет_гром_104.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ПРОЕКТ = os.path.dirname(КОРЕНЬ)
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро"), os.path.join(ПРОЕКТ, "scripts")]

from atoms_full103 import analyze_full_103  # noqa: E402
from оси import оси_звука  # noqa: E402
from фаза import загрузить  # noqa: E402
from ядро.пороги import WINDOW_MS_MAX, WINDOW_MS_MIN  # noqa: E402

SR_ATOM = 16000
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_grom.json")
WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "grom_real.wav")
OUT_CELL = os.path.join(КОРЕНЬ, "выход", "атомы_полные_гром", "etalon_grom_with_104.json")
OUT_PKG = os.path.join(КОРЕНЬ, "выход", "атомы_полные_гром", "атомы_звук_образ.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "обогащение_гром_params_104.md")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "обогащение_гром_params_104.json")


def _окно_ms(atom: dict) -> float:
    lt = float(atom.get("lifetime") or 0.05)
    return float(max(WINDOW_MS_MIN, min(WINDOW_MS_MAX, lt * 1000.0)))


def _окно_атома(x: np.ndarray, sr: int, atom: dict) -> np.ndarray:
    half = int(sr * _окно_ms(atom) / 2000.0)
    center = int(float(atom.get("birth") or 0) * sr + float(atom.get("lifetime") or 0) * sr * 0.5)
    i0 = max(0, center - half)
    i1 = min(len(x), center + half)
    chunk = x[i0:i1]
    min_len = max(sr // 50, 2048)
    if len(chunk) < min_len:
        chunk = np.pad(chunk, (0, min_len - len(chunk)))
    return chunk


def main() -> int:
    t0 = time.time()
    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms = cell.get("atoms") or cell.get("атомы") or []
    x, sr = загрузить(WAV)
    if sr != SR_ATOM:
        from scipy.signal import resample
        x = resample(x, int(len(x) * SR_ATOM / sr)).astype(np.float64)
        sr = SR_ATOM
    x = x / (float(np.max(np.abs(x)) or 1.0))

    n_ok = 0
    n_keys = 0
    for i, atom in enumerate(atoms):
        chunk = _окно_атома(x, sr, atom)
        p103 = analyze_full_103(chunk, sr)
        оси = оси_звука(chunk, sr)
        atom["params_104"] = {**p103, **оси}
        n_ok += 1
        n_keys = len(atom["params_104"])
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(atoms)}")

    cell["atoms"] = atoms
    if "атомы" in cell:
        cell["атомы"] = atoms
    cell["обогащение_104"] = {
        "дата": date.today().isoformat(),
        "источник_звук": "данные/клеточки/эталоны/grom_real.wav",
        "n_atoms": n_ok,
        "n_keys": n_keys,
        "эталон_json_не_тронут": True,
    }
    os.makedirs(os.path.dirname(OUT_CELL), exist_ok=True)
    with open(OUT_CELL, "w", encoding="utf-8") as f:
        json.dump(cell, f, ensure_ascii=False)

    if os.path.isfile(OUT_PKG):
        pkg = json.load(open(OUT_PKG, encoding="utf-8"))
        for i, fa in enumerate(pkg.get("atoms") or []):
            if i < len(atoms):
                fa["звук_ядро"]["params_104"] = atoms[i]["params_104"]
                fa["звук_ядро"]["долг_params_104"] = False
                fa["обратимость"]["params_104_на_атоме"] = True
        pkg["params_104_обогащение"] = cell["обогащение_104"]
        with open(OUT_PKG, "w", encoding="utf-8") as f:
            json.dump(pkg, f, ensure_ascii=False)

    elapsed = time.time() - t0
    report = {
        "дата": date.today().isoformat(),
        "n_atoms": n_ok,
        "n_keys_params_104": n_keys,
        "elapsed_s": round(elapsed, 1),
        "cell_copy": os.path.relpath(OUT_CELL, КОРЕНЬ),
        "sample": {
            "fd": atoms[100]["params_104"].get("fd"),
            "spectral_flatness": atoms[100]["params_104"].get("spectral_flatness"),
            "tonality": atoms[100]["params_104"].get("tonality"),
            "high_mid_freq_ratio": atoms[100]["params_104"].get("high_mid_freq_ratio"),
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(
            f"# Обогащение гром — params_104\n\n"
            f"> {report['дата']} · n={n_ok} · keys={n_keys} · {elapsed:.0f}s\n\n"
            f"- копия: `{os.path.relpath(OUT_CELL, КОРЕНЬ)}`\n"
            f"- пакет обновлён\n"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

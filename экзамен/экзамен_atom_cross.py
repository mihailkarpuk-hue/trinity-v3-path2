# -*- coding: utf-8 -*-
"""Экзамен atom+cross → sound → atom+cross (round-trip решётки).

  JSON (atoms+crosses) → гранулярный синтез → re-atomize → re-crosses → сверка.

Использование:
  python3 экзамен_atom_cross.py буквица_живая/живая_У_протяжное.json
  python3 экзамен_atom_cross.py --pilot   # У + О протяжное
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
КОРЕНЬ_V3 = os.path.dirname(КОРЕНЬ)
ПРОЕКТ = os.path.dirname(КОРЕНЬ_V3)
for p in (
    os.path.join(ПРОЕКТ, "scripts"),
    КОРЕНЬ_V3,
    os.path.join(КОРЕНЬ_V3, "ядро"),
    КОРЕНЬ,
):
    if p not in sys.path:
        sys.path.insert(0, p)

from round_trip_score import синтез as _синтез_legacy  # noqa: E402
from atoms_core28 import extract_atoms_28  # noqa: E402
from оси import оси_звука  # noqa: E402
from кресты import построить_кресты, сводка_решетки  # noqa: E402
from сшить_крест_клетки import предсказать_руки  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402

SR = 44100


def _load_audio(json_path: str) -> np.ndarray:
    from обогатить_атомы_104 import _load_any
    base = os.path.splitext(json_path)[0]
    for ext in (".m4a", ".wav", ".mp3"):
        p = base + ext
        if os.path.isfile(p):
            x, sr = _load_any(p)
            if sr != SR:
                from scipy.signal import resample
                x = resample(x, int(len(x) * SR / sr))
            return x.astype(np.float64)
    if "dozhd" in os.path.basename(json_path).lower():
        p = os.path.join(КОРЕНЬ_V3, "данные", "клеточки", "базис", "bazis_rain_dozhd.wav")
        if os.path.isfile(p):
            x, sr = _load_any(p)
            if sr != SR:
                from scipy.signal import resample
                x = resample(x, int(len(x) * SR / sr))
            return x.astype(np.float64)
    raise FileNotFoundError(f"нет звука для {json_path}")


def _synth(atoms: list[dict], crosses: list[dict] | None, rec: dict, dur: float | None = None) -> np.ndarray:
    meta = {**rec, "параметры104": rec.get("параметры104") or rec.get("parent_params_full")}
    return синтез_из_атомов(atoms, crosses, sr=SR, dur=dur, meta=meta)


def _atoms(rec: dict) -> list[dict]:
    return list(rec.get("atoms") or rec.get("атомы") or [])


def экзамен_json(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        rec = json.load(f)
    atoms0 = _atoms(rec)
    crosses0 = rec.get("crosses") or построить_кресты(atoms0)
    stats0 = сводка_решетки(atoms0, crosses0)

    x_orig = _load_audio(json_path)
    m = np.max(np.abs(x_orig))
    if m > 0:
        x_orig = x_orig / m
    оси_orig = оси_звука(x_orig[: min(len(x_orig), SR * 10)], SR)

    dur = rec.get("длительность_сек") or (len(x_orig) / SR)
    np.random.seed(0)
    y = _synth(atoms0, crosses0, rec, float(dur) + 0.1)
    atoms1, _ = extract_atoms_28(y, SR, os.path.basename(json_path))
    from обогатить_атомы_104 import обогатить_оси_из_звука
    обогатить_оси_из_звука(atoms1, y, SR)
    crosses1 = построить_кресты(atoms1)
    stats1 = сводка_решетки(atoms1, crosses1)
    оси_synth = оси_звука(y[: min(len(y), SR * 10)], SR)

    def ratio(a, b):
        return round(b / a, 3) if a else None

    stitch0 = предсказать_руки(atoms0, crosses0)
    stitch1 = предсказать_руки(atoms1, crosses1)

    return {
        "file": os.path.basename(json_path),
        "version": rec.get("version"),
        "params_104_per_atom": sum(1 for a in atoms0 if a.get("params_104")) / max(1, len(atoms0)),
        "atoms": {"in": len(atoms0), "out": len(atoms1), "ratio": ratio(len(atoms0), len(atoms1))},
        "crosses": {"in": len(crosses0), "out": len(crosses1), "ratio": ratio(len(crosses0), len(crosses1))},
        "axes_in": stats0.get("inter_edges_by_axis"),
        "axes_out": stats1.get("inter_edges_by_axis"),
        "fd": {"orig_audio": оси_orig["fd"], "synth_audio": оси_synth["fd"]},
        "nestedness": {"orig_audio": оси_orig["nestedness"], "synth_audio": оси_synth["nestedness"]},
        "stitch_верх": {
            "in": stitch0.get("верх_nestedness"),
            "out": stitch1.get("верх_nestedness"),
        },
        "stitch_низ": {
            "in": stitch0.get("низ_fd"),
            "out": stitch1.get("низ_fd"),
        },
        "coverage_harmonic": {
            "in": stats0["axis_coverage"]["harmonic"]["percent"],
            "out": stats1["axis_coverage"]["harmonic"]["percent"],
        },
    }


def _pilot():
    kl = os.path.join(КОРЕНЬ_V3, "данные", "клеточки")
    paths = [
        os.path.join(kl, "буквица_живая", "живая_У_протяжное.json"),
        os.path.join(kl, "клеточки_полные", "etalon_dozhd.json"),
    ]
    rows = []
    for p in paths:
        if os.path.isfile(p):
            r = экзамен_json(p)
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False, indent=2))
    out = os.path.join(КОРЕНЬ_V3, "данные", "экзамен_atom_cross_pilot.json")
    json.dump({"пилот": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n→ {out}")


def _all_bukvitsa():
    kl = os.path.join(КОРЕНЬ_V3, "данные", "клеточки", "буквица_живая")
    rows = []
    for name in sorted(os.listdir(kl)):
        if not name.startswith("живая_") or not name.endswith(".json"):
            continue
        p = os.path.join(kl, name)
        try:
            rows.append(экзамен_json(p))
        except Exception as e:
            rows.append({"file": name, "error": str(e)})
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pilot":
        _pilot()
    elif len(sys.argv) > 1 and sys.argv[1] == "--all-bukvitsa":
        rows = _all_bukvitsa()
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif len(sys.argv) > 1:
        jp = sys.argv[1]
        if not os.path.isabs(jp):
            cand = os.path.join(КОРЕНЬ_V3, jp)
            jp = cand if os.path.isfile(cand) else os.path.join(КОРЕНЬ_V3, "данные", "клеточки", jp)
        print(json.dumps(экзамен_json(jp), ensure_ascii=False, indent=2))
    else:
        print("usage: python3 экзамен_atom_cross.py <json> | --pilot")

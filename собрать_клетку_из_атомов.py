# -*- coding: utf-8 -*-
"""Сборка клетки atom-first: атомы (104+оси) + кресты → emergent cell → JSON.

Клетка не первична — складывается из целых атомов и сети крестов.
`параметры104` в JSON — производный агрегат (mean по атомам), не независимый анализ файла.

Использование:
  python3 собрать_клетку_из_атомов.py буквица_живая/живая_У_протяжное.json
  python3 собрать_клетку_из_атомов.py --pilot
  python3 собрать_клетку_из_атомов.py <json> --enrich
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from кресты import собрать_клетку  # noqa: E402

VERSION = "3.0-atom-first"


def _atoms_from_rec(rec: dict) -> list[dict]:
    return list(rec.get("atoms") or rec.get("атомы") or [])


def _find_audio(json_path: str, audio_path: str | None) -> str | None:
    if audio_path and os.path.isfile(audio_path):
        return audio_path
    base = os.path.splitext(json_path)[0]
    for ext in (".m4a", ".wav", ".mp3"):
        if os.path.isfile(base + ext):
            return base + ext
    if "dozhd" in os.path.basename(json_path).lower():
        cand = os.path.join(КОРЕНЬ, "данные", "клеточки", "базис", "bazis_rain_dozhd.wav")
        if os.path.isfile(cand):
            return cand
    return None


def _aggregate_p104(atoms: list[dict]) -> dict:
    if not atoms or not atoms[0].get("params_104"):
        return {}
    keys = [k for k in atoms[0]["params_104"] if isinstance(atoms[0]["params_104"][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [float(a["params_104"][k]) for a in atoms if k in a.get("params_104", {})]
        if vals:
            agg[k] = round(float(np.mean(vals)), 6)
    return agg


def собрать_из_json(
    json_path: str,
    *,
    audio_path: str | None = None,
    enrich: bool = False,
    in_place: bool = True,
) -> dict:
    """atoms + enrich(optional) + crosses → cell record v3.0-atom-first."""
    if enrich:
        from обогатить_атомы_104 import обогатить_json  # noqa: WPS433

        ap = _find_audio(json_path, audio_path)
        if not ap:
            raise FileNotFoundError(f"для --enrich нужен звук рядом с {json_path}")
        rec = обогатить_json(json_path, ap, in_place=in_place, force=True)
        atoms = _atoms_from_rec(rec)
        n_cross = rec.get("число_связей") or len(rec.get("crosses") or [])
        p104_n = len((atoms[0].get("params_104") or {})) if atoms else 0
        print(
            f"✓ {os.path.basename(json_path)} [enrich]: {len(atoms)} атомов · {n_cross} крестов · "
            f"104/атом={p104_n} · v={rec.get('version')}"
        )
        return rec

    with open(json_path, encoding="utf-8") as f:
        rec = json.load(f)

    atoms = _atoms_from_rec(rec)
    if not atoms:
        raise ValueError(f"нет atoms/атомы: {json_path}")

    t0 = time.time()
    cell = собрать_клетку(atoms, meta={
        "version": VERSION,
        "длительность_сек": rec.get("длительность_сек") or rec.get("parent_duration_sec"),
        "id": rec.get("id"),
    })
    rec["atoms"] = cell["atoms"]
    rec.pop("атомы", None)
    rec["crosses"] = cell["crosses"]
    rec["число_связей"] = cell["число_связей"]
    rec["решетка"] = cell.get("решетка")
    rec["version"] = VERSION
    rec["atoms_count"] = len(atoms)
    agg = _aggregate_p104(atoms)
    if agg:
        rec["параметры104"] = agg
        rec["parent_params_full"] = agg

    elapsed = time.time() - t0
    p104_n = len((atoms[0].get("params_104") or {})) if atoms else 0
    print(
        f"✓ {os.path.basename(json_path)}: {len(atoms)} атомов · {rec['число_связей']} крестов · "
        f"104/атом={p104_n} · v={VERSION} · {elapsed:.1f}s"
    )

    if in_place:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
    return rec


def _pilot(enrich: bool = False):
    paths = [
        os.path.join(КОРЕНЬ, "данные", "клеточки", "буквица_живая", "живая_У_протяжное.json"),
        os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json"),
    ]
    for p in paths:
        if os.path.isfile(p):
            собрать_из_json(p, enrich=enrich)


if __name__ == "__main__":
    enrich = "--enrich" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if "--pilot" in sys.argv:
        _pilot(enrich=enrich)
    elif args:
        jp = args[0]
        if not os.path.isabs(jp):
            cand = os.path.join(КОРЕНЬ, jp)
            jp = cand if os.path.isfile(cand) else os.path.join(КОРЕНЬ, "данные", "клеточки", jp)
        ap = args[1] if len(args) > 1 else None
        собрать_из_json(jp, audio_path=ap, enrich=enrich)
    else:
        print("usage: python3 собрать_клетку_из_атомов.py <json> [audio] | --pilot [--enrich]")

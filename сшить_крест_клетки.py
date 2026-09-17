# -*- coding: utf-8 -*-
"""Сшивка: решётка атомов → руки креста буквы vs агрегат клетки.

Уровень B (time/freq/harmonic) → предсказание уровня A (ЛЕВО/ПРАВО/ВЕРХ/НИЗ/ЦЕНТР).

  time-crosses      → pitch_fundamental  (ЛЕВО)
  freq-crosses      → spectral_centroid  (ПРАВО)
  harmonic-crosses  → nestedness         (ВЕРХ)
  noise_bridge      → fd                 (НИЗ)
  все атомы         → selfsim_r2, phase  (ЦЕНТР)

Использование:
  python3 сшить_крест_клетки.py буквица_живая/живая_У_протяжное.json
  python3 сшить_крест_клетки.py --all-bukvitsa
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from кресты import построить_кресты, сводка_решетки  # noqa: E402

КЛ = os.path.join(КОРЕНЬ, "данные", "клеточки")

# нормализация для сравнения разных шкал
NORM = {
    "лево": ("pitch_fundamental", 60, 400),
    "право": ("spectral_centroid", 200, 8000),
    "верх": ("nestedness", 0, 1),
    "низ": ("fd", 0, 1.2),
    "центр_r2": ("selfsim_r2", 0, 1),
}


def _фаза_атома(a: dict) -> str:
    ph = a.get("phase_gibbs") or a.get("phase_label")
    if ph in ("тон", "шум", "переход"):
        return ph
    h = float(a.get("harmonicity") or 0)
    if h > 0.6:
        return "тон"
    if h < 0.05:
        return "шум"
    return "переход"


def _p(atom: dict, key: str, default: float = 0.0) -> float:
    p = atom.get("params_104") or {}
    v = p.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _norm(val: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def предсказать_руки(atoms: list[dict], crosses: list[dict] | None = None) -> dict:
    crosses = crosses if crosses is not None else построить_кресты(atoms)
    if not atoms:
        return {}

    by_axis: dict[str, set[int]] = {"time": set(), "freq": set(), "harmonic": set()}
    noise_fds: list[float] = []
    w_time: list[tuple[float, float]] = []
    w_freq: list[tuple[float, float]] = []
    w_harm: list[tuple[float, float]] = []

    for c in crosses:
        a, b = int(c["atom_a"]), int(c["atom_b"])
        ax = c.get("axis", "")
        w = float(c.get("resonance") or 1)
        if ax in by_axis:
            by_axis[ax].update((a, b))
        if ax == "time":
            w_time.append((_p(atoms[a], "pitch_fundamental"), w))
            w_time.append((_p(atoms[b], "pitch_fundamental"), w))
        if ax == "freq":
            w_freq.append((_p(atoms[a], "spectral_centroid"), w))
            w_freq.append((_p(atoms[b], "spectral_centroid"), w))
        if ax == "harmonic":
            w_harm.append((_p(atoms[a], "nestedness"), w))
            w_harm.append((_p(atoms[b], "nestedness"), w))
        if c.get("cross_type") == "noise_bridge":
            noise_fds.extend([_p(atoms[a], "fd"), _p(atoms[b], "fd")])
        elif _фаза_атома(atoms[a]) == "шум" or _фаза_атома(atoms[b]) == "шум":
            noise_fds.extend([_p(atoms[a], "fd"), _p(atoms[b], "fd")])

    def wmean(pairs: list[tuple[float, float]]) -> float:
        if not pairs:
            return 0.0
        ws = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / ws if ws else 0.0

    def mean_atoms(idxs: set[int], key: str) -> float:
        if not idxs:
            return 0.0
        vals = [_p(atoms[i], key) for i in idxs if i < len(atoms)]
        return float(np.mean(vals)) if vals else 0.0

    лево = wmean(w_time) or mean_atoms(by_axis["time"], "pitch_fundamental")
    право = wmean(w_freq) or mean_atoms(by_axis["freq"], "spectral_centroid")
    верх = wmean(w_harm) or mean_atoms(by_axis["harmonic"], "nestedness")
    if noise_fds:
        низ = float(np.mean(noise_fds))
    else:
        shum = [i for i, a in enumerate(atoms) if _фаза_атома(a) == "шум"]
        низ = mean_atoms(set(shum), "fd") if shum else float(np.mean([_p(a, "fd") for a in atoms]))

    r2_vals = [_p(a, "selfsim_r2") for a in atoms]
    phases = [_фаза_атома(a) for a in atoms]
    from collections import Counter
    dom_phase = Counter(phases).most_common(1)[0][0] if phases else "переход"

    return {
        "лево_f0": round(лево, 4),
        "право_centroid": round(право, 4),
        "верх_nestedness": round(верх, 4),
        "низ_fd": round(низ, 4),
        "центр_phase": dom_phase,
        "центр_r2": round(float(np.mean(r2_vals)), 4),
        "crosses_used": len(crosses),
        "решетка": сводка_решетки(atoms, crosses),
    }


def агрегат_клетки(rec: dict) -> dict:
    p = rec.get("параметры104") or rec.get("parent_params_full") or {}
    return {
        "лево_f0": float(p.get("pitch_fundamental") or 0),
        "право_centroid": float(p.get("spectral_centroid") or 0),
        "верх_nestedness": float(p.get("nestedness") or 0),
        "низ_fd": float(p.get("fd") or 0),
        "центр_r2": float(p.get("selfsim_r2") or 0),
    }


def сравнить(pred: dict, agg: dict) -> dict:
    out = {}
    pairs = [
        ("лево", "лево_f0", "лево_f0"),
        ("право", "право_centroid", "право_centroid"),
        ("верх", "верх_nestedness", "верх_nestedness"),
        ("низ", "низ_fd", "низ_fd"),
        ("центр_r2", "центр_r2", "центр_r2"),
    ]
    for name, pk, ak in pairs:
        pv, av = pred.get(pk, 0), agg.get(ak, 0)
        key_norm, lo, hi = NORM.get(name, (name, 0, 1))
        pn, an = _norm(pv, lo, hi), _norm(av, lo, hi)
        out[name] = {
            "predicted": pv,
            "aggregate": av,
            "delta_norm": round(abs(pn - an), 4),
        }
    out["mean_delta_norm"] = round(float(np.mean([out[k]["delta_norm"] for k in out if k != "mean_delta_norm"])), 4)
    return out


def обработать_json(path: str) -> dict | None:
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    atoms = rec.get("atoms") or []
    if not atoms or not atoms[0].get("params_104"):
        return None
    crosses = rec.get("crosses") or построить_кресты(atoms)
    pred = предсказать_руки(atoms, crosses)
    agg = агрегат_клетки(rec)
    cmp = сравнить(pred, agg)
    return {"file": os.path.basename(path), "predicted": pred, "aggregate": agg, "compare": cmp}


def _all_bukvitsa():
    paths = sorted(glob.glob(os.path.join(КЛ, "буквица_живая", "живая_*.json")))
    results = []
    for p in paths:
        r = обработать_json(p)
        if r:
            results.append(r)
            print(f"  {r['file']}: Δnorm={r['compare']['mean_delta_norm']:.3f} "
                  f"верх {r['compare']['верх']['predicted']:.3f}→{r['compare']['верх']['aggregate']:.3f} "
                  f"низ {r['compare']['низ']['predicted']:.3f}→{r['compare']['низ']['aggregate']:.3f}")
        else:
            print(f"  skip (нет params_104): {os.path.basename(p)}")
    if len(results) >= 2:
        for arm in ("лево", "право", "верх", "низ", "центр_r2"):
            preds = [r["compare"][arm]["predicted"] for r in results]
            aggs = [r["compare"][arm]["aggregate"] for r in results]
            if np.std(preds) > 1e-9 and np.std(aggs) > 1e-9:
                r = float(np.corrcoef(preds, aggs)[0, 1])
                print(f"  r({arm}) = {r:.3f}  n={len(results)}")
    out = os.path.join(КОРЕНЬ, "данные", "сшивка_крест_буквица.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": results, "count": len(results)}, f, ensure_ascii=False, indent=2)
    print(f"→ {out}")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all-bukvitsa":
        _all_bukvitsa()
    elif len(sys.argv) > 1:
        jp = sys.argv[1]
        if not os.path.isabs(jp):
            cand = os.path.join(КОРЕНЬ, jp)
            jp = cand if os.path.isfile(cand) else os.path.join(КЛ, jp)
        r = обработать_json(jp)
        if not r:
            print("нет params_104 — сначала обогатить_атомы_104.py")
            sys.exit(1)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print("usage: python3 сшить_крест_клетки.py <json> | --all-bukvitsa")

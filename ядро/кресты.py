# -*- coding: utf-8 -*-
"""кресты.py — решётка 3×4 (канон = atoms_lattice_3cross.py).

Уровень B: межатомные связи по осям time / freq / harmonic.
Мост к уровню A (крест буквы): time→лево, freq→право, harmonic→верх.

См. Тринити cursor/СПРАВОЧНИК §3, данные/КРЕСТЫ_МАППИНГ.md
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


INTERNAL_EDGES = "time-freq;freq-harmonic;harmonic-time"


def _f0(atom: dict) -> float:
    f = float(atom.get("freq") or 0)
    hi = int(atom.get("harmonic_index") or 0)
    return f / max(1, hi) if hi > 0 else f


def _resonance(a: dict, b: dict, axis: str) -> float:
    if axis == "time":
        dt = abs(float(a.get("birth", 0)) - float(b.get("birth", 0)))
        return round(max(0.0, 1.0 - dt / 0.08), 4)
    if axis == "freq":
        fa, fb = float(a.get("freq") or 1), float(b.get("freq") or 1)
        if fa <= 0 or fb <= 0:
            return 0.0
        ratio = max(fa, fb) / min(fa, fb)
        if ratio < 1.06:
            return 0.95
        if abs(math.log2(ratio) - round(math.log2(ratio))) < 0.08:
            return 0.75
        return round(max(0.0, 0.5 - abs(math.log2(ratio)) * 0.15), 4)
    if axis == "harmonic":
        f0a, f0b = _f0(a), _f0(b)
        if f0a <= 0 or f0b <= 0:
            return 0.0
        rel = abs(f0a - f0b) / f0a
        return round(max(0.0, 1.0 - rel / 0.05), 4)
    return 0.5


def _atom_phase(a: dict) -> str:
    h = float(a.get("harmonicity") or 0)
    if h > 0.6:
        return "тон"
    if h < 0.05:
        return "шум"
    return "переход"


def _cross_type(a: dict, b: dict) -> str:
    pa, pb = _atom_phase(a), _atom_phase(b)
    if pa != pb:
        return "phase_transition"
    if pa == "шум" or float(a.get("noisiness_local") or 0) > 0.5:
        return "noise_bridge"
    return "tonal_link"


def _master_face(axis: str, cross_type: str) -> str:
    if cross_type == "phase_transition":
        return "центр"
    return {"time": "лево", "freq": "право", "harmonic": "верх"}.get(axis, "центр")


def _energy_flow(a: dict, b: dict, axis: str, direction: str) -> str:
    if axis == "time":
        ba, bb = float(a.get("birth", 0)), float(b.get("birth", 0))
        if abs(ba - bb) < 1e-6:
            return "mutual"
        return "A→B" if ba < bb else "B→A"
    if axis == "freq":
        return "A→B" if direction == "upper" else "B→A"
    if axis == "harmonic":
        return "A→B" if direction == "up" else "B→A"
    return "mutual"


def построить_кресты(atoms: list[dict], *, f0_tol: float = 0.05) -> list[dict]:
    """Решётка 3cross — один ребро на пару, порядок: time → freq → harmonic."""
    N = len(atoms)
    if N < 2:
        return []

    crosses: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add(i: int, j: int, axis: str, direction: str):
        if i == j:
            return
        pair = (min(i, j), max(i, j))
        if pair in seen:
            return
        a, b = atoms[i], atoms[j]
        seen.add(pair)
        ct = _cross_type(a, b)
        crosses.append({
            "atom_a": i,
            "atom_b": j,
            "axis": axis,
            "direction": direction,
            "type": "inter",
            "resonance": _resonance(a, b, axis),
            "cross_type": ct,
            "energy_flow": _energy_flow(a, b, axis, direction),
            "master_cross_face": _master_face(axis, ct),
        })

    # TIME: цепочка по birth (как lattice + проекция_3d)
    order_t = sorted(range(N), key=lambda i: float(atoms[i].get("birth", 0)))
    for k in range(N - 1):
        add(order_t[k], order_t[k + 1], "time", "next")

    # FREQ: цепочка по freq
    order_f = sorted(range(N), key=lambda i: float(atoms[i].get("freq", 0)))
    for k in range(N - 1):
        add(order_f[k], order_f[k + 1], "freq", "upper")

    # HARMONIC: hi ±1, f0 ±5%
    by_hi: dict[int, list[int]] = {}
    for i in range(N):
        by_hi.setdefault(int(atoms[i].get("harmonic_index") or 0), []).append(i)
    for i in range(N):
        hi = int(atoms[i].get("harmonic_index") or 0)
        f0_i = _f0(atoms[i])
        for delta in (-1, 1):
            target = hi + delta
            if target < 0 or target not in by_hi:
                continue
            best_j, best_rel = None, float("inf")
            for j in by_hi[target]:
                if j == i:
                    continue
                f0_j = _f0(atoms[j])
                rel = abs(f0_i - f0_j) / f0_i if f0_i > 0 else abs(f0_i - f0_j)
                if rel < f0_tol and rel < best_rel:
                    best_rel, best_j = rel, j
            if best_j is not None:
                add(i, best_j, "harmonic", "up" if delta == 1 else "down")

    crosses.sort(key=lambda c: (-c["resonance"], c["atom_a"], c["atom_b"]))
    return crosses


def сводка_решетки(atoms: list[dict], crosses: list[dict] | None = None) -> dict[str, Any]:
    """Аналог summarize_lattice_3cross."""
    crosses = crosses if crosses is not None else построить_кресты(atoms)
    N = len(atoms)
    by_axis = Counter(c["axis"] for c in crosses)
    deg: dict[int, int] = {i: 0 for i in range(N)}
    for c in crosses:
        deg[c["atom_a"]] += 1
        deg[c["atom_b"]] += 1
    degrees = list(deg.values())

    def axis_coverage(axis: str) -> dict:
        covered = {c["atom_a"] for c in crosses if c["axis"] == axis} | {
            c["atom_b"] for c in crosses if c["axis"] == axis
        }
        return {"atoms_covered": len(covered), "percent": round(len(covered) / N * 100, 1) if N else 0}

    return {
        "nodes": N,
        "internal_edges_per_atom": 3,
        "inter_edges_total": len(crosses),
        "inter_edges_by_axis": dict(by_axis),
        "atoms_full_connectivity_ge6": sum(1 for d in degrees if d >= 6),
        "atoms_isolated_0": sum(1 for d in degrees if d == 0),
        "avg_inter_degree": round(sum(degrees) / N, 2) if N else 0,
        "axis_coverage": {ax: axis_coverage(ax) for ax in ("time", "freq", "harmonic")},
    }


def обогатить_внутренние_кресты(atoms: list[dict]) -> None:
    for a in atoms:
        a["internal_edges"] = INTERNAL_EDGES


def собрать_клетку(atoms: list[dict], meta: dict | None = None) -> dict[str, Any]:
    обогатить_внутренние_кресты(atoms)
    crosses = построить_кресты(atoms)
    return {
        "atoms": atoms,
        "crosses": crosses,
        "решетка": сводка_решетки(atoms, crosses),
        "число_атомов": len(atoms),
        "число_связей": len(crosses),
        **(meta or {}),
    }

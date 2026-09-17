# -*- coding: utf-8 -*-
"""Матрица узнавания: top-5 соседей для 55 обогащённых клеток по метрике d₉.

usage: python3 scripts/матрица_узнавания.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)

from ядро.метрика import d9, d9_по_группам, загрузить_индекс  # noqa: E402

КАТАЛОГ = os.path.join(КОРЕНЬ, "данные", "клеточки", "каталог.json")
ОТЧЁТ_JSON = os.path.join(КОРЕНЬ, "отчёты", "матрица_узнавания_БАЗА.json")
ОТЧЁТ_MD = os.path.join(КОРЕНЬ, "отчёты", "матрица_узнавания_БАЗА.md")
ФОРМУЛА = os.path.join(КОРЕНЬ, "данные", "формула_калибровки.json")

def _обогащённая(c: dict) -> bool:
    """Клетка с per-atom params_104 (55 шт. по ТЗ)."""
    path = c.get("атомы") or c.get("решётка")
    if not path:
        return False
    fp = os.path.join(КОРЕНЬ, "данные", "клеточки", path)
    if not os.path.isfile(fp):
        return False
    try:
        d = json.load(open(fp, encoding="utf-8"))
        atoms = d.get("atoms") or []
        if not atoms:
            return False
        p = atoms[0].get("params_104") or {}
        return len(p) >= 100
    except (json.JSONDecodeError, OSError):
        return False


def _вектор(c: dict, ключи: list[str]) -> list[float] | None:
    п = c.get("параметры104") or {}
    if len(п) < 100:
        return None
    return [float(п.get(k) or 0.0) for k in ключи]


def epsilon_порог(клетки: list[dict], ключи: list[str], mu, sd) -> float:
    """Медиана d₉ между разными клетками одной группы каталога."""
    by_grp: dict[str, list[tuple[str, list[float]]]] = {}
    for c in клетки:
        v = _вектор(c, ключи)
        if v is None:
            continue
        g = c.get("группа") or "?"
        by_grp.setdefault(g, []).append((c["id"], v))

    dists = []
    for g, items in by_grp.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                dists.append(d9(items[i][1], items[j][1], mu=mu, sd=sd))
    if not dists:
        return 0.0
    return float(np.median(dists))


def main() -> None:
    os.makedirs(os.path.dirname(ОТЧЁТ_JSON), exist_ok=True)
    idx = загрузить_индекс()
    ключи = idx["оси"]
    mu, sd = idx["mu"], idx["sd"]

    кат = json.load(open(КАТАЛОГ, encoding="utf-8"))
    все = next(v for v in кат.values() if isinstance(v, list))
    обогащ = [c for c in все if _обогащённая(c) and _вектор(c, ключи)]

    # индекс всех клеток для kNN
    все_век = []
    for c in все:
        v = _вектор(c, ключи)
        if v:
            все_век.append({
                "id": c["id"],
                "название": c.get("название") or c["id"],
                "группа": c.get("группа") or "",
                "вектор": v,
            })

    матрица = []
    for c in обогащ:
        q = _вектор(c, ключи)
        assert q is not None
        соседи = []
        for other in все_век:
            if other["id"] == c["id"]:
                continue
            d = d9(q, other["вектор"], mu=mu, sd=sd)
            соседи.append({**other, "d": round(d, 4)})
        соседи.sort(key=lambda x: x["d"])
        top5 = [{k: s[k] for k in ("id", "название", "группа", "d")} for s in соседи[:5]]
        row = {
            "id": c["id"],
            "название": c.get("название") or c["id"],
            "группа": c.get("группа") or "",
            "top5": top5,
        }
        if top5:
            t1 = next(x for x in все_век if x["id"] == top5[0]["id"])
            row["группы_d9_top1"] = {
                k: round(v, 4) for k, v in d9_по_группам(q, t1["вектор"], mu=mu, sd=sd).items()
            }
        матрица.append(row)

    eps = epsilon_порог(все, ключи, mu, sd)

    out = {
        "версия": 1,
        "метрика": "d9",
        "epsilon_порог": round(eps, 6),
        "число_обогащённых": len(обогащ),
        "матрица": матрица,
    }
    with open(ОТЧЁТ_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # обновить формула_калибровки.json
    формула = json.load(open(ФОРМУЛА, encoding="utf-8"))
    формула["epsilon_порог"] = round(eps, 6)
    формула["epsilon_метод"] = "медиана d9 между разными клетками одной группы каталога"
    with open(ФОРМУЛА, "w", encoding="utf-8") as f:
        json.dump(формула, f, ensure_ascii=False, indent=2)
        f.write("\n")

    lines = [
        "# Матрица узнавания — база (d₉)",
        "",
        f"Обогащённых клеток: **{len(обогащ)}** · ε_порог = **{eps:.4f}**",
        "",
        "| клетка | группа | top-1 | d | top-2 | d |",
        "|---|---|---|---|---|",
    ]
    for row in матрица:
        t = row["top5"]
        t0 = t[0] if t else {"название": "—", "d": 0}
        t1 = t[1] if len(t) > 1 else {"название": "—", "d": 0}
        lines.append(
            f"| {row['название']} | {row['группа']} | {t0['название']} | {t0['d']:.3f} "
            f"| {t1['название']} | {t1['d']:.3f} |"
        )
    with open(ОТЧЁТ_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✓ {os.path.relpath(ОТЧЁТ_JSON, КОРЕНЬ)} · {len(обогащ)} клеток")
    print(f"✓ ε_порог = {eps:.6f} → формула_калибровки.json")


if __name__ == "__main__":
    main()

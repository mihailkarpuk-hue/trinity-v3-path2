# -*- coding: utf-8 -*-
"""Пилот: FULL vs синтез_104 на 10 буквах — nestedness, fd, готовность 104.

  python3 экзамен/экзамен_104_pilot.py
  python3 экзамен/экзамен_104_pilot.py --all
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(os.path.dirname(КОРЕНЬ), "scripts"), КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро")):
    if p not in sys.path:
        sys.path.insert(0, p)

from оси import оси_звука  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402
from синтез_104 import синтез_104_из_атомов, статистика_готовности  # noqa: E402

SR = 44100
KL = os.path.join(КОРЕНЬ, "данные", "клеточки", "буквица_живая")
ПИЛОТ = ["Л", "У_протяжное", "О_протяжное", "М", "Тт", "Е_протяжная", "Ю_протяжное", "С_протяжное", "А", "Ф"]


def _names(all_: bool) -> list[str]:
    if not all_:
        return ПИЛОТ
    out = []
    for fn in sorted(os.listdir(KL)):
        if fn.startswith("живая_") and fn.endswith(".json"):
            out.append(fn[6:-5])
    return out


def _оси(y):
    return оси_звука(y[: min(len(y), SR * 10)], SR)


def main(all_: bool = False):
    rows = []
    for name in _names(all_):
        rec = json.load(open(os.path.join(KL, f"живая_{name}.json"), encoding="utf-8"))
        atoms = rec.get("atoms") or []
        crosses = rec.get("crosses")
        meta = {**rec, "параметры104": rec.get("параметры104") or rec.get("parent_params_full")}
        dur = float(rec.get("длительность_сек") or 2) + 0.1
        st = статистика_готовности(atoms)
        np.random.seed(0)
        y_full = синтез_из_атомов(atoms, crosses, sr=SR, dur=dur, meta=meta)
        y_104 = синтез_104_из_атомов(atoms, crosses, sr=SR, dur=dur, meta=meta)
        o_full, o_104 = _оси(y_full), _оси(y_104)
        rows.append({
            "буква": name,
            "p104_pct": st["pct"],
            "FULL_nest": o_full["nestedness"],
            "V104_nest": o_104["nestedness"],
            "FULL_fd": o_full["fd"],
            "V104_fd": o_104["fd"],
        })
        print(f"{name:16s} p104={st['pct']:5.1f}%  nest FULL={o_full['nestedness']:.3f} 104={o_104['nestedness']:.3f}  fd {o_full['fd']:.3f}/{o_104['fd']:.3f}")

    out = os.path.join(КОРЕНЬ, "данные", "экзамен_104_pilot.json")
    json.dump({"строки": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main("--all" in sys.argv)

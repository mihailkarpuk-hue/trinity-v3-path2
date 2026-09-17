# -*- coding: utf-8 -*-
"""Реальные цифры живого звука и его образа — БЕЗ нормировки.
Для каждой клетки: ОБРАЗ (атомы: birth/freq/amp/harmonicity/lifetime — реальные)
и ЗВУК (108 сырых параметров фенотипа, сгруппированы). Ничего не нормируем,
ничего не режем. Только измеряем и записываем.
Использование: python3 scripts/реальные_цифры.py живая_А живая_Б живая_В
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import добыча_словаря as Д  # noqa
from ядро.ключи_108 import KEYS_BY_GROUP  # noqa

ЦЕЛИ = sys.argv[1:] or ["живая_А", "живая_Б", "живая_В"]
ВЫХОД = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "отчёты", "реальные_цифры.md",
)


def дамп(cid: str, out: list) -> None:
    c = Д._cell(cid)
    ap = Д._audio_path(c)
    x, sr = Д._load_audio(ap)
    dur = len(x) / sr
    # ОБРАЗ: реальные атомы
    jp = os.path.join(Д.КЛЕТКИ, c.get("атомы") or "")
    atoms = json.load(open(jp, encoding="utf-8")).get("atoms") or []
    # ЗВУК: сырые 108 (без нормировки)
    ph = Д._фенотип(x, sr)

    out.append(f"\n## {cid}   ·   длительность {dur:.3f} с   ·   атомов {len(atoms)}\n")
    out.append("### ОБРАЗ — атомы (реальные величины)\n")
    out.append("| # | birth,с | freq,Гц | amp | harmonicity | lifetime,с | фаза |")
    out.append("|---|---|---|---|---|---|---|")
    for i, a in enumerate(atoms):
        out.append(
            f"| {i} | {a.get('birth',0):.4f} | {a.get('freq',0):.1f} | "
            f"{a.get('amp',0):.4f} | {a.get('harmonicity',0):.3f} | "
            f"{a.get('lifetime',0):.4f} | {a.get('phase','')} |"
        )
    out.append("\n### ЗВУК — 108 сырых параметров (без нормировки)\n")
    for g, keys in KEYS_BY_GROUP.items():
        vals = "  ·  ".join(f"{k}={ph.get(k, 0):.4g}" for k in keys)
        out.append(f"**{g}** ({len(keys)}): {vals}\n")


def main() -> None:
    out = ["# Реальные цифры живого звука и образа (без нормировки)\n"]
    for cid in ЦЕЛИ:
        try:
            дамп(cid, out)
        except Exception as e:  # noqa
            out.append(f"\n## {cid} — ОШИБКА: {type(e).__name__}: {e}\n")
    txt = "\n".join(out)
    os.makedirs(os.path.dirname(ВЫХОД), exist_ok=True)
    open(ВЫХОД, "w", encoding="utf-8").write(txt)
    # краткая сводка в консоль
    for cid in ЦЕЛИ:
        c = Д._cell(cid)
        x, sr = Д._load_audio(Д._audio_path(c))
        ph = Д._фенотип(x, sr)
        jp = os.path.join(Д.КЛЕТКИ, c.get("атомы") or "")
        atoms = json.load(open(jp, encoding="utf-8")).get("atoms") or []
        print(f"{cid}: атомов={len(atoms)}  centroid={ph.get('spectral_centroid',0):.0f}Гц  "
              f"flatness={ph.get('spectral_flatness',0):.3f}  "
              f"pitch={ph.get('pitch_fundamental',0):.0f}Гц  "
              f"F1={ph.get('formant_f1',0):.0f}  F2={ph.get('formant_f2',0):.0f}  "
              f"harmonic_ratio={ph.get('harmonic_ratio',0):.3f}")
    print(f"\n→ полный дамп: {ВЫХОД}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Вшить кресты клетки etalon_dozhd в пакет полных атомов.

Не пересобираем топологию — копируем crosses (+ решётка) как есть,
добавляем id_a/id_b пакета и компактное соседство на каждом атоме.

Запуск: python3 scripts/вшить_кресты_в_пакет_дождь.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import date

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")
PKG = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "атомы_звук_образ.json")
META = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "meta.json")
OUT_CROSSES = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "кресты.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "кресты_в_пакете_дождь.md")


def main() -> int:
    cell = json.load(open(CELL, encoding="utf-8"))
    pkg = json.load(open(PKG, encoding="utf-8"))
    atoms = pkg["atoms"]
    n = len(atoms)
    if n != len(cell.get("atoms") or []):
        raise SystemExit(f"рассинхрон атомов пакет={n} клетка={len(cell['atoms'])}")

    raw = cell.get("crosses") or []
    if not raw:
        raise SystemExit("в клетке нет crosses")

    crosses = []
    adj = defaultdict(list)  # atom_idx -> list of compact edge refs
    for i, c in enumerate(raw):
        ia = int(c["atom_a"])
        ib = int(c["atom_b"])
        if not (0 <= ia < n and 0 <= ib < n):
            continue
        id_a = atoms[ia]["id"]
        id_b = atoms[ib]["id"]
        edge = {
            "i": i,
            "atom_a": ia,
            "atom_b": ib,
            "id_a": id_a,
            "id_b": id_b,
            "axis": c.get("axis"),
            "direction": c.get("direction"),
            "type": c.get("type"),
            "resonance": c.get("resonance"),
            "cross_type": c.get("cross_type"),
            "energy_flow": c.get("energy_flow"),
            "master_cross_face": c.get("master_cross_face"),
        }
        crosses.append(edge)
        compact = {
            "cross_i": i,
            "axis": edge["axis"],
            "resonance": edge["resonance"],
            "cross_type": edge["cross_type"],
            "face": edge["master_cross_face"],
            "direction": edge["direction"],
            "energy_flow": edge["energy_flow"],
        }
        adj[ia].append({**compact, "к": id_b, "к_idx": ib})
        adj[ib].append({**compact, "к": id_a, "к_idx": ia})

    for i, a in enumerate(atoms):
        links = adj.get(i) or []
        a["кресты"] = {
            "n": len(links),
            "связи": links,
        }
        obr = a.get("обратимость") or {}
        obr["кресты_на_атоме"] = True
        a["обратимость"] = obr

    # sidecar (удобно грузить отдельно) + в корне пакета
    side = {
        "дата": date.today().isoformat(),
        "клетка": "etalon_dozhd",
        "источник": "данные/клеточки/клеточки_полные/etalon_dozhd.json#crosses",
        "n_atoms": n,
        "n_crosses": len(crosses),
        "число_связей": len(crosses),
        "решетка": cell.get("решетка"),
        "пересобраны": False,
        "note": "топология клетки без изменений; id_a/id_b = ids пакета",
        "crosses": crosses,
    }
    with open(OUT_CROSSES, "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False)

    pkg["дата"] = date.today().isoformat()
    pkg["crosses"] = crosses
    pkg["число_связей"] = len(crosses)
    pkg["решетка"] = cell.get("решетка")
    pkg["crosses_ref"] = os.path.relpath(OUT_CROSSES, КОРЕНЬ)
    pkg["crosses_в_пакете"] = True
    pkg["crosses_мета"] = {
        "дата": date.today().isoformat(),
        "n_crosses": len(crosses),
        "пересобраны": False,
        "источник_клетка": "etalon_dozhd",
        "sidecar": os.path.relpath(OUT_CROSSES, КОРЕНЬ),
        "на_каждом_атоме": "кресты.связи",
    }
    with open(PKG, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False)

    meta = {}
    if os.path.isfile(META):
        meta = json.load(open(META, encoding="utf-8"))
    meta.update({
        "дата": date.today().isoformat(),
        "n_crosses": len(crosses),
        "crosses_в_пакете": True,
        "crosses_file": os.path.relpath(OUT_CROSSES, КОРЕНЬ),
    })
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # stats
    from collections import Counter
    axes = Counter(c["axis"] for c in crosses)
    ctypes = Counter(c["cross_type"] for c in crosses)
    degrees = [atoms[i]["кресты"]["n"] for i in range(n)]
    report_lines = [
        "# Кресты в пакете полных атомов (дождь)",
        "",
        f"> {date.today().isoformat()} · n_crosses=**{len(crosses)}** · на каждом атоме `кресты.связи`",
        "",
        "## Вердикт",
        "",
        "- Топология **не пересобиралась** — копия из `etalon_dozhd`.",
        "- В корне пакета: `crosses[]`, `число_связей`, `решетка`.",
        f"- Sidecar: `{os.path.relpath(OUT_CROSSES, КОРЕНЬ)}`",
        "- На атоме: `кресты.n` + `кресты.связи[]` (к, axis, resonance, cross_type, face).",
        "",
        "## Статистика",
        "",
        f"| ось | n |",
        f"|-----|--:|",
    ]
    for k, v in axes.most_common():
        report_lines.append(f"| {k} | {v} |")
    report_lines += [
        "",
        "| cross_type | n |",
        "|------------|--:|",
    ]
    for k, v in ctypes.most_common():
        report_lines.append(f"| {k} | {v} |")
    report_lines += [
        "",
        f"- степень: min={min(degrees)} med={sorted(degrees)[len(degrees)//2]} max={max(degrees)}",
        f"- атомов со связями: {sum(1 for d in degrees if d > 0)} / {n}",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(json.dumps({
        "ok": True,
        "n_crosses": len(crosses),
        "n_atoms": n,
        "degree_med": sorted(degrees)[len(degrees) // 2],
        "sidecar": os.path.relpath(OUT_CROSSES, КОРЕНЬ),
        "pkg_mb": round(os.path.getsize(PKG) / 1e6, 2),
        "crosses_mb": round(os.path.getsize(OUT_CROSSES) / 1e6, 2),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

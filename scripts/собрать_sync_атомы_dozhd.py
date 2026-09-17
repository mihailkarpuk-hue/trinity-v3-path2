# -*- coding: utf-8 -*-
"""Собрать пакет sync-атомов дождь — режим A (кусок B№2).

Берёт равномерную выборку из раскадровка_dozhd_A.json,
подтягивает звук_ядро из клетки etalon_dozhd (первый атом с тем же birth),
пишет выход/sync_atoms_dozhd_A.json по схеме_sync_атом_dozhd.md.

Запуск: python3 scripts/собрать_sync_атомы_dozhd.py
"""
from __future__ import annotations

import json
import os
from datetime import date

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
РАСК = os.path.join(КОРЕНЬ, "отчёты", "раскадровка_dozhd_A.json")
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")
OUT = os.path.join(КОРЕНЬ, "выход", "sync_atoms_dozhd_A.json")
MD = os.path.join(КОРЕНЬ, "отчёты", "пакет_sync_атомы_dozhd_A.md")
N = 12

ЗВУК = "данные/клеточки/эталоны/dozhd_real.wav"
CLEAN = "данные/стихии_живые/dozhd/clean_video.mp4"

SOUND_KEYS = (
    "freq",
    "amp",
    "phase",
    "lifetime",
    "harmonicity",
    "spectral_width",
    "centroid_local",
    "brightness_local",
    "noisiness_local",
    "pos_x",
    "pos_y",
    "pos_z",
    "color_r",
    "color_g",
    "color_b",
)


def _pick(rows: list, n: int) -> list:
    if len(rows) <= n:
        return list(rows)
    idxs = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    seen, out = set(), []
    for i in idxs:
        if i in seen:
            continue
        seen.add(i)
        out.append(rows[i])
    return out


def _index_by_birth(atoms: list) -> dict[float, list]:
    idx: dict[float, list] = {}
    for i, a in enumerate(atoms):
        b = round(float(a.get("birth") or 0), 6)
        idx.setdefault(b, []).append(i)
    return idx


def main() -> int:
    rask = json.load(open(РАСК, encoding="utf-8"))
    rows = rask.get("rows") or []
    pick = _pick(rows, N)

    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    cell_atoms = cell.get("atoms") or []
    by_birth = _index_by_birth(cell_atoms)

    sync = []
    for r in pick:
        birth = round(float(r["birth"]), 6)
        frame_i = int(r["frame_i"])
        refs = by_birth.get(birth) or []
        if not refs:
            # ближайший birth
            keys = sorted(by_birth.keys(), key=lambda b: abs(b - birth))
            refs = by_birth[keys[0]] if keys else []
            birth_used = keys[0] if keys else birth
        else:
            birth_used = birth

        src = cell_atoms[refs[0]] if refs else {}
        sound = {k: src.get(k) for k in SOUND_KEYS if k in src}
        sound["atom_ref"] = {
            "клетка": "etalon_dozhd",
            "index": refs[0] if refs else None,
            "birth": birth_used,
            "n_at_birth": len(refs),
        }
        sound["params_104"] = src.get("params_104")
        sound["долг_params_104"] = src.get("params_104") is None

        atom = {
            "id": f"dozhd_A_{birth}_{frame_i}",
            "пилот": "dozhd",
            "mode": "A",
            "alignment": "assumed_t0",
            "birth": birth,
            "frame_i": frame_i,
            "t_frame": r.get("t_frame"),
            "звук_путь": ЗВУК,
            "clean_video": CLEAN,
            "образ_кусочек": {
                "path": None,
                "frame_i": frame_i,
                "источник": "clean_video",
                "примечание": "crop в куске B№3",
            },
            "звук_ядро": sound,
            "обратимость": {
                "звук_в_петле": True,
                "образ_в_синтезе": False,
                "образ_привязан": True,
                "note": "path кадра null до B№3",
            },
        }
        sync.append(atom)

    packet = {
        "дата": date.today().isoformat(),
        "схема": "отчёты/схема_sync_атом_dozhd.md",
        "mode": "A",
        "alignment": "assumed_t0",
        "n": len(sync),
        "atoms": sync,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(packet, f, ensure_ascii=False, indent=2)

    lines = [
        "# Пакет sync-атомов дождь A",
        "",
        f"> {packet['дата']} · N={len(sync)} · path образа пока null",
        "",
        f"- файл: `{os.path.relpath(OUT, КОРЕНЬ)}`",
        "",
        "| id | birth | frame_i | freq | долг_104 |",
        "|----|------:|--------:|-----:|:--------:|",
    ]
    for a in sync:
        s = a["звук_ядро"]
        lines.append(
            f"| `{a['id']}` | {a['birth']} | {a['frame_i']} | {s.get('freq')} | {s.get('долг_params_104')} |"
        )
    lines.append("")
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"ok": True, "n": len(sync), "out": OUT}, ensure_ascii=False))
    return 0 if len(sync) >= 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Пересборка 55 обогащённых клеток атомизацией V2 (треки).

usage: python3 scripts/пересборка_треков.py [--dry-run] [id_клетки...]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from ядро.атомизация import атомизировать  # noqa: E402
from ядро.фаза import загрузить  # noqa: E402
from ядро.кресты import собрать_клетку  # noqa: E402

КЛЕТКИ = os.path.join(КОРЕНЬ, "данные", "клеточки")
КАТАЛОГ = os.path.join(КЛЕТКИ, "каталог.json")
БЭКАП = os.path.join(КЛЕТКИ, "_BAK_треки")


def _обогащённая(c: dict) -> bool:
    path = c.get("атомы") or c.get("решётка")
    if not path:
        return False
    fp = os.path.join(КЛЕТКИ, path)
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


def _визуал(atom: dict, i: int, n: int) -> dict:
    """Минимальные pos/color для браузера из физики атома."""
    import colorsys

    f = float(atom.get("freq") or 440)
    a = float(atom.get("amp") or 0.5)
    b = float(atom.get("birth") or 0)
    t = np.log10(max(80, min(f, 9000)) / 80) / np.log10(9000 / 80)
    hue = 270 - t * 270
    sat = 0.35 + float(atom.get("harmonicity") or 0) * 0.55
    r, g, bl = colorsys.hls_to_rgb((hue % 360) / 360, 0.5, sat)
    return {
        "pos_x": round(b * 3.0 - 1.0, 3),
        "pos_y": round(float(i) / max(n - 1, 1) * 2 - 1, 3),
        "pos_z": round(a * 0.8 - 0.2, 3),
        "color_r": int(r * 255),
        "color_g": int(g * 255),
        "color_b": int(bl * 255),
        "size": round(0.08 + a * 0.2, 3),
    }


def _load_audio(path: str):
    if path.lower().endswith(".wav"):
        return загрузить(path)
    from обогатить_атомы_104 import _load_any  # noqa: WPS433
    return _load_any(path)


def _audio_path(c: dict) -> str | None:
    z = c.get("звук") or ""
    p = os.path.join(КЛЕТКИ, z)
    if os.path.isfile(p):
        return p
    atoms_path = c.get("атомы") or ""
    base = os.path.join(КЛЕТКИ, os.path.splitext(atoms_path)[0])
    for ext in (".wav", ".m4a", ".mp3"):
        if os.path.isfile(base + ext):
            return base + ext
    return None


def пересобрать_json(json_path: str, audio_path: str, *, dry_run: bool = False) -> dict:
    x, sr = _load_audio(audio_path)
    raw = атомизировать(x, sr)
    n = len(raw)
    atoms = []
    for i, a in enumerate(raw):
        atom = {k: v for k, v in a.items() if not k.startswith("_")}
        atom.update(_визуал(atom, i, n))
        atoms.append(atom)

    cell = собрать_клетку(atoms, meta={"version": "3.1-tracks"})
    rec = {
        "version": "3.1-tracks",
        "длительность_сек": round(len(x) / sr, 4),
        "atoms_count": len(atoms),
        "atoms": cell["atoms"],
        "crosses": cell["crosses"],
        "число_связей": cell["число_связей"],
    }

    if dry_run:
        return rec

    os.makedirs(БЭКАП, exist_ok=True)
    if os.path.isfile(json_path):
        bak = os.path.join(БЭКАП, os.path.basename(json_path))
        if not os.path.isfile(bak):
            shutil.copy2(json_path, bak)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)

    # обогащение 104
    from обогатить_атомы_104 import обогатить_json  # noqa: WPS433

    return обогатить_json(json_path, audio_path, force=True)


def main() -> None:
    dry = "--dry-run" in sys.argv
    ids = [a for a in sys.argv[1:] if not a.startswith("-")]

    cat = json.load(open(КАТАЛОГ, encoding="utf-8"))
    cells = next(v for v in cat.values() if isinstance(v, list))
    targets = [c for c in cells if _обогащённая(c)]
    if ids:
        targets = [c for c in targets if c["id"] in ids]

    t0 = time.time()
    ok = 0
    for c in targets:
        ap = _audio_path(c)
        jp = os.path.join(КЛЕТКИ, c.get("атомы") or "")
        if not ap or not os.path.isfile(jp):
            print(f"✗ {c['id']}: нет звука/json")
            continue
        try:
            rec = пересобрать_json(jp, ap, dry_run=dry)
        except Exception as e:
            print(f"✗ {c['id']}: {e}")
            continue
        n = rec.get("atoms_count") or len(rec.get("atoms") or [])
        kap = " ✓" if c["id"] == "bazis_12_kaplya" and n <= 12 else ""
        print(f"✓ {c['id']}: {n} атомов{kap}")
        ok += 1

    print(f"Готово: {ok}/{len(targets)} за {time.time()-t0:.0f}с · бэкап: {БЭКАП}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8
"""Пересборка atom JSON для буквицы_живая из m4a/wav (свежая атомизация V2).

Обновляет atoms + crosses + параметры104 в каталоге. Бэкап: данные/клеточки/_BAK_треки/

usage:
  python3 scripts/пересборка_буквицы.py
  python3 scripts/пересборка_буквицы.py --dry-run
  python3 scripts/пересборка_буквицы.py --force живая_А
"""
from __future__ import annotations

import json
import os
import sys
import time

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.join(КОРЕНЬ, "scripts"))

from пересборка_треков import (  # noqa: E402
    _audio_path,
    _визуал,
    _обогащённая,
    БЭКАП,
)

КЛЕТКИ = os.path.join(КОРЕНЬ, "данные", "клеточки")
КАТАЛОГ = os.path.join(КЛЕТКИ, "каталог.json")
SR = 22050


def _load_audio_sr22050(path: str):
    """Тот же путь, что CP#2 / добыча_словаря (22050 Hz)."""
    from scripts.добыча_словаря import _load_audio  # noqa: WPS433
    return _load_audio(path, SR)


def пересобрать_букву(json_path: str, audio_path: str, *, dry_run: bool = False) -> dict:
    import shutil
    from ядро.атомизация import атомизировать  # noqa: WPS433
    from ядро.кресты import собрать_клетку  # noqa: WPS433

    x, sr = _load_audio_sr22050(audio_path)
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
    from обогатить_атомы_104 import обогатить_json  # noqa: WPS433
    return обогатить_json(json_path, audio_path, force=True)


def _bukvitsa_cells() -> list[dict]:
    cat = json.load(open(КАТАЛОГ, encoding="utf-8"))
    cells = next(v for v in cat.values() if isinstance(v, list))
    return [c for c in cells if c.get("группа") == "буквица_живая"]


def main() -> None:
    dry = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    selective = "--selective" in sys.argv or ("--all" not in sys.argv and not force)
    ids = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = [c for c in _bukvitsa_cells() if _обогащённая(c)]
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
            old_n = len(json.load(open(jp, encoding="utf-8")).get("atoms") or [])
            x, sr = _load_audio_sr22050(ap)
            from ядро.атомизация import атомизировать  # noqa: WPS433
            fresh_n = len(атомизировать(x, sr))
            if selective and not force and fresh_n <= old_n and old_n < 80:
                print(f"· {c['id']}: {old_n} атомов (fresh={fresh_n}, пропуск)")
                ok += 1
                continue
            if selective and old_n >= 80:
                print(f"! {c['id']}: подозрительно {old_n} атомов → fresh={fresh_n}")
            rec = пересобрать_букву(jp, ap, dry_run=dry)
            n = rec.get("atoms_count") or len(rec.get("atoms") or [])
            print(f"✓ {c['id']}: {old_n} → {n} атомов (fresh={fresh_n})")
            ok += 1
        except Exception as e:
            print(f"✗ {c['id']}: {e}")

    print(f"Готово: {ok}/{len(targets)} за {time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()

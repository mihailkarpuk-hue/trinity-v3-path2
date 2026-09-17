# -*- coding: utf-8 -*-
"""Вырезать образ_кусочек для sync-атомов дождь A (кусок B№3).

Читает выход/sync_atoms_dozhd_A.json, кадры из clean_video →
выход/sync_atoms_dozhd_A/crops/, обновляет path в json.

Запуск: python3 scripts/вырезать_sync_crops_dozhd.py
"""
from __future__ import annotations

import json
import os
from datetime import date

import cv2

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKET = os.path.join(КОРЕНЬ, "выход", "sync_atoms_dozhd_A.json")
CLEAN = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "clean_video.mp4")
CROPS = os.path.join(КОРЕНЬ, "выход", "sync_atoms_dozhd_A", "crops")
MD = os.path.join(КОРЕНЬ, "отчёты", "crops_sync_атомы_dozhd_A.md")


def main() -> int:
    data = json.load(open(PACKET, encoding="utf-8"))
    atoms = data.get("atoms") or []
    if len(atoms) < 12:
        raise SystemExit(f"мало атомов в пакете: {len(atoms)}")

    cap = cv2.VideoCapture(CLEAN)
    if not cap.isOpened():
        raise SystemExit(f"не открыть {CLEAN}")

    os.makedirs(CROPS, exist_ok=True)
    written = []
    for a in atoms:
        fi = int(a["frame_i"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise SystemExit(f"нет кадра {fi}")
        name = f"{a['id']}.png".replace(".", "p")
        # id already has dots in birth — safer name:
        name = f"f{fi:04d}_b{float(a['birth']):.3f}".replace(".", "p") + ".png"
        path = os.path.join(CROPS, name)
        cv2.imwrite(path, frame)
        rel = os.path.relpath(path, КОРЕНЬ)
        a["образ_кусочек"]["path"] = rel
        a["образ_кусочек"]["frame_i"] = fi
        a["обратимость"]["образ_привязан"] = True
        a["обратимость"]["note"] = "crop на диске; в синтез звука не входит"
        written.append({"id": a["id"], "path": rel, "frame_i": fi})
    cap.release()

    data["дата_crops"] = date.today().isoformat()
    with open(PACKET, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    lines = [
        "# Crops sync-атомов дождь A",
        "",
        f"> {data['дата_crops']} · N={len(written)}",
        "",
        f"- папка: `{os.path.relpath(CROPS, КОРЕНЬ)}`",
        f"- пакет обновлён: `{os.path.relpath(PACKET, КОРЕНЬ)}`",
        "",
        "| id | frame_i | path |",
        "|----|--------:|------|",
    ]
    for w in written:
        lines.append(f"| `{w['id']}` | {w['frame_i']} | `{w['path']}` |")
    lines.append("")
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"ok": True, "n": len(written), "crops": CROPS}, ensure_ascii=False))
    return 0 if len(written) >= 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())

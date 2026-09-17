# -*- coding: utf-8 -*-
"""Превью раскадровки глазу — пилот дождь режим A.

Читает отчёты/раскадровка_dozhd_A.json, достаёт кадры из clean_video,
рисует метку birth/frame_i, пишет png + index.html.

Запуск: python3 scripts/превью_раскадровка_синхрон.py
"""
from __future__ import annotations

import json
import os
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_IN = os.path.join(КОРЕНЬ, "отчёты", "раскадровка_dozhd_A.json")
CLEAN = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "clean_video.mp4")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "раскадровка_dozhd_A")
MD_OUT = os.path.join(КОРЕНЬ, "отчёты", "превью_раскадровка_dozhd_A.md")
N_PREVIEW = 12


def _pick_rows(rows: list, n: int) -> list:
    if len(rows) <= n:
        return list(rows)
    idxs = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    seen = set()
    out = []
    for i in idxs:
        if i in seen:
            continue
        seen.add(i)
        out.append(rows[i])
    return out


def _label(frame: np.ndarray, birth: float, frame_i: int) -> np.ndarray:
    img = frame.copy()
    text = f"birth={birth:.3f}s  frame={frame_i}  A/assumed_t0"
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return img


def main() -> int:
    data = json.load(open(JSON_IN, encoding="utf-8"))
    rows = data.get("rows") or []
    if not rows:
        raise SystemExit("пустая раскадровка")

    pick = _pick_rows(rows, N_PREVIEW)
    cap = cv2.VideoCapture(CLEAN)
    if not cap.isOpened():
        raise SystemExit(f"не открыть {CLEAN}")

    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for r in pick:
        fi = int(r["frame_i"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        labeled = _label(frame, float(r["birth"]), fi)
        name = f"f{fi:04d}_b{float(r['birth']):.3f}".replace(".", "p") + ".png"
        path = os.path.join(OUT_DIR, name)
        cv2.imwrite(path, labeled)
        written.append({"file": name, "birth": r["birth"], "frame_i": fi})
    cap.release()

    if len(written) < 8:
        raise SystemExit(f"мало кадров: {len(written)}")

    # index.html
    cards = "\n".join(
        f'<figure><img src="{w["file"]}" alt="f{w["frame_i"]}"/>'
        f"<figcaption>birth={w['birth']} · frame={w['frame_i']}</figcaption></figure>"
        for w in written
    )
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>Раскадровка дождь A</title>
<style>
body{{margin:0;background:#111;color:#ddd;font:14px/1.4 system-ui,sans-serif}}
h1{{font-size:18px;padding:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;padding:16px}}
figure{{margin:0;background:#1a1a1a}}
img{{width:100%;display:block}}
figcaption{{padding:8px;font-size:12px;opacity:.85}}
.meta{{padding:0 16px 16px;opacity:.7;font-size:12px}}
</style></head><body>
<h1>Превью раскадровки — дождь · режим A</h1>
<p class="meta">alignment=assumed_t0 · {date.today().isoformat()} · не доказанный синхрон · обратимость = кусок №6</p>
<div class="grid">{cards}</div>
</body></html>
"""
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    rel = os.path.relpath(OUT_DIR, КОРЕНЬ)
    lines = [
        "# Превью раскадровки глазу — дождь A",
        "",
        f"> {date.today().isoformat()} · mode=A · alignment=assumed_t0",
        "",
        f"- кадров в превью: **{len(written)}** (из {len(rows)} строк таблицы)",
        f"- папка: `{rel}/`",
        f"- открыть: `file://…/{rel}/index.html`",
        "",
        "| файл | birth | frame_i |",
        "|------|------:|--------:|",
    ]
    for w in written:
        lines.append(f"| `{w['file']}` | {w['birth']} | {w['frame_i']} |")
    lines += [
        "",
        "**Заметка:** это глаз на clean-кадрах по таблице k3, не экзамен обратимости.",
        "",
    ]
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"ok": True, "n": len(written), "dir": OUT_DIR}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Вырезать ТОЛЬКО каплю — малый ROI из событий drop_only."""
from __future__ import annotations

import json
import os
from datetime import date

import cv2

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.json")
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "crops_drop")
MD = os.path.join(КОРЕНЬ, "отчёты", "crops_только_капля.md")
HTML = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "E_только_капля.html")


def main() -> int:
    data = json.load(open(EVENTS, encoding="utf-8"))
    events = data.get("events") or []
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit("no video")
    os.makedirs(OUT_DIR, exist_ok=True)

    # очистить старые широкие crops смысла «пейзаж» не трогаем — новая папка
    written = []
    for ev in events:
        fi = int(ev["frame_i"])
        x, y, bw, bh = ev["bbox"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        crop = frame[y : y + bh, x : x + bw]
        if crop.size == 0:
            continue
        # увеличить для глаза (nearest), не добавляя пейзаж
        scale = max(1, 192 // max(bw, bh))
        big = cv2.resize(crop, (bw * scale, bh * scale), interpolation=cv2.INTER_NEAREST)
        # крест в центре = «вот атом-капля»
        cx, cy = big.shape[1] // 2, big.shape[0] // 2
        cv2.drawMarker(big, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(
            big,
            f"{ev['id']} {ev['t_sec']:.2f}s",
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        name = f"{ev['id']}_t{ev['t_sec']:.2f}.png"
        path = os.path.join(OUT_DIR, name)
        cv2.imwrite(path, big)
        rel = os.path.relpath(path, КОРЕНЬ)
        ev["crop_path"] = rel
        written.append(ev)
    cap.release()

    with open(EVENTS, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    lines = [
        "# Только капля — малый отрывок мироздания",
        "",
        f"> {date.today().isoformat()} · N={len(written)} · без листвы/неба в ROI",
        "",
        "| id | t_sec | crop_side | path |",
        "|----|------:|----------:|------|",
    ]
    figs = []
    for ev in written:
        side = ev.get("crop_side") or ev["bbox"][2]
        lines.append(f"| `{ev['id']}` | {ev['t_sec']} | {side} | `{ev['crop_path']}` |")
        figs.append(
            f"<figure><img src=\"crops_drop/{os.path.basename(ev['crop_path'])}\"/>"
            f"<figcaption>{ev['id']} · t={ev['t_sec']}s · {side}px</figcaption></figure>"
        )
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    html = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"/>
<title>E — только капля</title>
<style>
body{{margin:0;background:#0a0a0a;color:#ddd;font:14px system-ui}}
h1{{padding:16px 16px 4px}}
.meta{{padding:0 16px 16px;opacity:.8;max-width:640px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;padding:16px}}
figure{{margin:0;background:#111;border:1px solid #333}}
img{{width:100%;image-rendering:pixelated;display:block}}
figcaption{{padding:6px;font-size:11px}}
</style></head><body>
<h1>Отрывок мироздания = одна капля</h1>
<p class="meta">Маленький квадрат вокруг капли (не листвы, не облака, не берег).
Жёлтый крест — центр детекции. Скажи: это капля? ок / нет / править.</p>
<div class="grid">{''.join(figs)}</div>
</body></html>"""
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps({"ok": True, "n": len(written), "html": HTML}, ensure_ascii=False))
    return 0 if len(written) >= 5 else 2


if __name__ == "__main__":
    raise SystemExit(main())

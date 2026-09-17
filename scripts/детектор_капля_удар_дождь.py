# -*- coding: utf-8 -*-
"""Детектор ОДНОЙ капли (крошечный кусок мироздания) на video_live_01.

Не «кусок пейзажа», а маленький объект: полоска/точка капли или корона удара.
ROI фиксированного малого размера вокруг центра капли.

Запуск: python3 scripts/детектор_капля_удар_дождь.py
"""
from __future__ import annotations

import json
import os
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.md")

MAX_SEC = 25.0
MIN_GAP_SEC = 0.25
TOP_N = 16
# зона лужи (ниже листвы/неба)
Y0_FRAC = 0.42
Y1_FRAC = 0.95
X0_FRAC = 0.08
X1_FRAC = 0.92
# размер «отрывка» капли — маленький квадрат
CROP = 56
# контур капли: не пейзаж
MIN_AREA = 8
MAX_AREA = 900  # выше = уже кусок сцены, отбраковать
MAX_ASPECT = 8.0  # полоска капли может быть вытянутой


def _crop_box(cx: int, cy: int, side: int, w: int, h: int) -> list[int]:
    half = side // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    if x0 + side > w:
        x0 = max(0, w - side)
    if y0 + side > h:
        y0 = max(0, h - side)
    return [int(x0), int(y0), int(min(side, w - x0)), int(min(side, h - y0))]


def main() -> int:
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"не открыть {VIDEO}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_f = int(MAX_SEC * fps)

    x0 = int(w * X0_FRAC)
    x1 = int(w * X1_FRAC)
    y0 = int(h * Y0_FRAC)
    y1 = int(h * Y1_FRAC)

    ok, prev = cap.read()
    if not ok:
        raise SystemExit("пустой клип")
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    candidates = []  # (score, frame_i, cx, cy, area, aspect)
    fi = 0
    while fi < max_f:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(g, prev_g)
        prev_g = g

        roi = d[y0:y1, x0:x1]
        # капли/всплески — яркие мелкие пятна движения
        _, th = cv2.threshold(roi, 22, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < MIN_AREA or area > MAX_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > MAX_ASPECT:
                continue
            # яркость движения внутри контура
            mask = np.zeros(roi.shape, np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            score = float(cv2.mean(roi, mask=mask)[0]) * (area ** 0.5)
            cx = x0 + x + bw // 2
            cy = y0 + y + bh // 2
            if best is None or score > best[0]:
                best = (score, fi, cx, cy, area, aspect)
        if best is not None:
            candidates.append(best)
    cap.release()

    if not candidates:
        raise SystemExit("нет мелких кандидатов-капель — смени порог/клип")

    # пики по времени: оставляем локально сильные, с разнесением
    candidates.sort(key=lambda t: t[0], reverse=True)
    gap = max(1, int(MIN_GAP_SEC * fps))
    chosen = []
    used = []
    for score, fi, cx, cy, area, aspect in candidates:
        if any(abs(fi - uf) < gap for uf in used):
            continue
        # не у самого края
        if cx < 20 or cy < 20 or cx > w - 20 or cy > h - 20:
            continue
        used.append(fi)
        chosen.append((score, fi, cx, cy, area, aspect))
        if len(chosen) >= TOP_N:
            break
    chosen.sort(key=lambda t: t[1])

    smax = max(t[0] for t in chosen) or 1.0
    events = []
    for i, (score, fi, cx, cy, area, aspect) in enumerate(chosen):
        bbox = _crop_box(cx, cy, CROP, w, h)
        events.append(
            {
                "id": f"drop_{i:03d}",
                "событие": "drop_only",
                "video": "данные/стихии_живые/dozhd/video_live_01.mp4",
                "frame_i": int(fi),
                "t_sec": round(fi / fps, 4),
                "cx": int(cx),
                "cy": int(cy),
                "bbox": bbox,
                "crop_side": CROP,
                "area_contur": round(area, 1),
                "aspect": round(aspect, 2),
                "фаза": "drop",
                "score": round(float(score / smax), 4),
                "звук": "native_aac",
                "правило": "только капля — малый ROI, без листвы/неба как содержимого атома",
            }
        )

    report = {
        "дата": date.today().isoformat(),
        "video": os.path.relpath(VIDEO, КОРЕНЬ),
        "fps": fps,
        "прогон_сек": MAX_SEC,
        "crop_side_px": CROP,
        "n_events": len(events),
        "events": events,
        "note": "отрывок мироздания = одна капля (малый квадрат), не кусок сцены",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Кандидаты: только капля (малый ROI)",
        "",
        f"> {report['дата']} · crop={CROP}px · N={len(events)}",
        "",
        "| id | t_sec | frame | cx,cy | area | score |",
        "|----|------:|------:|-------|-----:|------:|",
    ]
    for ev in events:
        lines.append(
            f"| `{ev['id']}` | {ev['t_sec']} | {ev['frame_i']} | {ev['cx']},{ev['cy']} | {ev['area_contur']} | {ev['score']} |"
        )
    lines += ["", f"JSON: `{os.path.relpath(OUT_JSON, КОРЕНЬ)}`", ""]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"ok": True, "n": len(events), "crop": CROP}, ensure_ascii=False))
    return 0 if len(events) >= 5 else 2


if __name__ == "__main__":
    raise SystemExit(main())

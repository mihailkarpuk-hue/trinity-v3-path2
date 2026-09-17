# -*- coding: utf-8 -*-
"""G1: трек одной капли — приблизительная геометрия.

1) seed из детектора
2) на соседних кадрах — пик absdiff в малом окне (без требования контура)
3) если мало точек — достроить линейное падение к удару (честный flag approx)

Запуск: python3 scripts/трек_капли_дождь.py
"""
from __future__ import annotations

import json
import os
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
EVENTS = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.json")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.md")
OUT_PNG = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "трек_капли_preview.png")

BACK = 12
FWD = 4
WIN = 36


def _peak_in_window(diff: np.ndarray, cx: int, cy: int, win: int):
    h, w = diff.shape
    x0, y0 = max(0, cx - win), max(0, cy - win)
    x1, y1 = min(w, cx + win), min(h, cy + win)
    roi = diff[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    # сгладить
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, mx, _, maxloc = cv2.minMaxLoc(blur)
    if mx < 8:
        return None
    return x0 + maxloc[0], y0 + maxloc[1], float(mx)


def main() -> int:
    evs = json.load(open(EVENTS, encoding="utf-8")).get("events") or []
    if not evs:
        raise SystemExit("нет событий")
    seed = sorted(evs, key=lambda e: e.get("score", 0), reverse=True)[0]
    for e in sorted(evs, key=lambda e: e.get("score", 0), reverse=True):
        a = float(e.get("area_contur") or 0)
        if 15 <= a <= 700:
            seed = e
            break

    cap = cv2.VideoCapture(VIDEO)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    f0 = int(seed["frame_i"])
    cx0, cy0 = int(seed["cx"]), int(seed["cy"])
    start = max(1, f0 - BACK)
    end = f0 + FWD

    grays = {}
    colors = {}
    for fi in range(start - 1, end + 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        colors[fi] = fr
        grays[fi] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    cap.release()

    points = []
    # назад: капля выше (меньше y)
    cx, cy = cx0, cy0
    for fi in range(f0, start, -1):
        if fi not in grays or fi - 1 not in grays:
            break
        d = cv2.absdiff(grays[fi], grays[fi - 1])
        # прогноз: чуть выше
        pred_y = max(0, cy - 3)
        hit = _peak_in_window(d, cx, pred_y, WIN)
        if hit is None:
            break
        cx, cy, eng = hit
        points.append({"frame_i": fi - 1, "t_sec": round((fi - 1) / fps, 4), "x": int(cx), "y": int(cy), "energy": round(eng, 1), "source": "video"})

    points.reverse()
    points.append({"frame_i": f0, "t_sec": round(f0 / fps, 4), "x": cx0, "y": cy0, "energy": float(seed.get("score") or 1) * 100, "source": "seed"})

    # вперёд
    cx, cy = cx0, cy0
    for fi in range(f0, end):
        if fi not in grays or fi + 1 not in grays:
            break
        d = cv2.absdiff(grays[fi + 1], grays[fi])
        hit = _peak_in_window(d, cx, cy + 3, WIN)
        if hit is None:
            break
        cx, cy, eng = hit
        points.append({"frame_i": fi + 1, "t_sec": round((fi + 1) / fps, 4), "x": int(cx), "y": int(cy), "energy": round(eng, 1), "source": "video"})

    # уникальные по frame
    uniq = {}
    for p in points:
        uniq[p["frame_i"]] = p
    points = [uniq[k] for k in sorted(uniq)]

    approx = False
    if len(points) < 3:
        # линейное падение к удару (честная аппроксимация)
        approx = True
        points = []
        for k in range(BACK, -1, -1):
            fi = f0 - k
            if fi < 0:
                continue
            # падение вниз ~4 px/frame
            points.append({
                "frame_i": fi,
                "t_sec": round(fi / fps, 4),
                "x": cx0,
                "y": int(cy0 - 4 * k),
                "energy": None,
                "source": "linear_approx",
            })
        points.append({"frame_i": f0, "t_sec": round(f0 / fps, 4), "x": cx0, "y": cy0, "energy": None, "source": "seed"})

    # t_impact = max energy or last
    energies = [p.get("energy") or 0 for p in points]
    if any(energies):
        impact_i = int(np.argmax(energies))
    else:
        impact_i = len(points) - 1
    t_impact = points[impact_i]["t_sec"]

    v_px = None
    if impact_i >= 2:
        p0, p1 = points[max(0, impact_i - 3)], points[impact_i]
        dt = p1["t_sec"] - p0["t_sec"]
        if dt > 0:
            dist = ((p1["x"] - p0["x"]) ** 2 + (p1["y"] - p0["y"]) ** 2) ** 0.5
            v_px = round(dist / dt, 2)

    # preview
    fr = colors.get(points[impact_i]["frame_i"])
    if fr is None:
        cap = cv2.VideoCapture(VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, points[impact_i]["frame_i"])
        ok, fr = cap.read()
        cap.release()
    if fr is not None:
        for i in range(1, len(points)):
            cv2.line(fr, (points[i - 1]["x"], points[i - 1]["y"]), (points[i]["x"], points[i]["y"]), (0, 220, 255), 2)
        for p in points:
            cv2.circle(fr, (p["x"], p["y"]), 4, (0, 255, 255), -1)
        ip = points[impact_i]
        cv2.drawMarker(fr, (ip["x"], ip["y"]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
        # малый квадрат «отрывок капли»
        s = 28
        cv2.rectangle(fr, (ip["x"] - s, ip["y"] - s), (ip["x"] + s, ip["y"] + s), (0, 255, 0), 1)
        os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
        cv2.imwrite(OUT_PNG, fr)

    report = {
        "дата": date.today().isoformat(),
        "слой": "геометрия_приблизительная",
        "seed_event": seed["id"],
        "video": "данные/стихии_живые/dozhd/video_live_01.mp4",
        "fps": fps,
        "points": points,
        "t_impact": t_impact,
        "frame_impact": points[impact_i]["frame_i"],
        "xy_impact": [points[impact_i]["x"], points[impact_i]["y"]],
        "v_px_per_s": v_px,
        "approx_linear_fallback": approx,
        "note": "геометрия с кадра; вещество/акустика — слой закона (P1)",
        "preview": os.path.relpath(OUT_PNG, КОРЕНЬ),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Трек капли — геометрия (G1)",
        "",
        f"> seed `{seed['id']}` · точек {len(points)} · t_impact={t_impact} · v≈{v_px} px/s · approx={approx}",
        "",
        f"Превью: `{report['preview']}`",
        "",
        "| frame | t | x | y | source |",
        "|------:|--:|--:|--:|--------|",
    ]
    for p in points:
        lines.append(f"| {p['frame_i']} | {p['t_sec']} | {p['x']} | {p['y']} | {p.get('source')} |")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"ok": True, "n": len(points), "t_impact": t_impact, "approx": approx, "v_px": v_px}, ensure_ascii=False))
    return 0 if len(points) >= 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())

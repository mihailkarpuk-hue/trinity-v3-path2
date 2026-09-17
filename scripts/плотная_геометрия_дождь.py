# -*- coding: utf-8 -*-
"""Плотная геометрия капель на video_live_01 (для xy в атомах).

Отличие от детектора TOP-16:
- несколько пятен на кадр (не одно)
- мелкий временной шаг + spatial NMS
- окно вокруг geometry_t0 (birth+offset) + полный прогон для запаса
- короткие треки назад от сильных ударов

Запуск: python3 scripts/плотная_геометрия_дождь.py
"""
from __future__ import annotations

import json
import os
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
ALIGN = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "alignment_t0.json")
OUT_EVENTS = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.json")
OUT_EVENTS_MD = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.md")
OUT_TRACK = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.json")
OUT_TRACK_MD = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.md")
OUT_DENSE = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "геометрия_плотная.json")
OUT_PREVIEW = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "геометрия_плотная_preview.jpg")

# ROI лужи
Y0_FRAC, Y1_FRAC = 0.40, 0.96
X0_FRAC, X1_FRAC = 0.05, 0.95

MIN_AREA, MAX_AREA = 6, 700
MAX_ASPECT = 7.0
THR = 18
PER_FRAME = 8
SPATIAL_NMS_PX = 28
MIN_GAP_SEC = 0.04  # между событиями в одной «ячейке» пространства
TOP_EVENTS = 400
TRACK_BACK = 10
TRACK_WIN = 32
CROP = 48


def _nms_spatial(blobs, min_dist: float):
    """blobs: list of (score, cx, cy, area, aspect)"""
    blobs = sorted(blobs, key=lambda t: t[0], reverse=True)
    kept = []
    for b in blobs:
        if any((b[1] - k[1]) ** 2 + (b[2] - k[2]) ** 2 < min_dist ** 2 for k in kept):
            continue
        kept.append(b)
    return kept


def _peak_in_window(diff: np.ndarray, cx: int, cy: int, win: int):
    h, w = diff.shape
    x0, y0 = max(0, cx - win), max(0, cy - win)
    x1, y1 = min(w, cx + win), min(h, cy + win)
    roi = diff[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, mx, _, maxloc = cv2.minMaxLoc(blur)
    if mx < 7:
        return None
    return x0 + maxloc[0], y0 + maxloc[1], float(mx)


def detect_dense(cap, fps, w, h, t0: float, t1: float):
    x0 = int(w * X0_FRAC)
    x1 = int(w * X1_FRAC)
    y0 = int(h * Y0_FRAC)
    y1 = int(h * Y1_FRAC)
    f_start = max(1, int(t0 * fps))
    f_end = int(t1 * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start - 1)
    ok, prev = cap.read()
    if not ok:
        return []
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    fi = f_start - 1
    raw = []  # score, fi, cx, cy, area, aspect

    while fi < f_end:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(g, prev_g)
        prev_g = g
        roi = d[y0:y1, x0:x1]
        _, th = cv2.threshold(roi, THR, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < MIN_AREA or area > MAX_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > MAX_ASPECT:
                continue
            mask = np.zeros(roi.shape, np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            score = float(cv2.mean(roi, mask=mask)[0]) * (area ** 0.45)
            cx = x0 + x + bw // 2
            cy = y0 + y + bh // 2
            if cx < 16 or cy < 16 or cx > w - 16 or cy > h - 16:
                continue
            blobs.append((score, cx, cy, area, aspect))
        for score, cx, cy, area, aspect in _nms_spatial(blobs, SPATIAL_NMS_PX)[:PER_FRAME]:
            raw.append((score, fi, int(cx), int(cy), area, aspect))
    return raw


def select_events(raw, fps, top_n: int):
    raw = sorted(raw, key=lambda t: t[0], reverse=True)
    gap = max(1, int(MIN_GAP_SEC * fps))
    chosen = []
    # temporal+spatial occupancy: (fi_bin, cell)
    used = []
    cell = 40
    for score, fi, cx, cy, area, aspect in raw:
        key = (fi // gap, cx // cell, cy // cell)
        if any(abs(fi - uf) < gap and abs(cx - ux) < cell and abs(cy - uy) < cell for uf, ux, uy in used):
            continue
        # softer: also reject exact same cell recently
        if key in {(uf // gap, ux // cell, uy // cell) for uf, ux, uy in used[-200:]}:
            # allow if score much stronger — skip for density
            pass
        used.append((fi, cx, cy))
        chosen.append((score, fi, cx, cy, area, aspect))
        if len(chosen) >= top_n:
            break
    chosen.sort(key=lambda t: t[1])
    return chosen


def track_from_seed(video, seed, fps):
    """Короткий трек назад/вперёд от seed. Возвращает points list."""
    cap = cv2.VideoCapture(video)
    f0 = int(seed["frame_i"])
    cx0, cy0 = int(seed["cx"]), int(seed["cy"])
    start = max(1, f0 - TRACK_BACK)
    end = f0 + 3
    grays = {}
    for fi in range(start - 1, end + 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        grays[fi] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    cap.release()

    points = []
    cx, cy = cx0, cy0
    for fi in range(f0, start, -1):
        if fi not in grays or fi - 1 not in grays:
            break
        d = cv2.absdiff(grays[fi], grays[fi - 1])
        hit = _peak_in_window(d, cx, max(0, cy - 3), TRACK_WIN)
        if hit is None:
            break
        cx, cy, eng = hit
        points.append({
            "frame_i": fi - 1,
            "t_sec": round((fi - 1) / fps, 4),
            "x": int(cx),
            "y": int(cy),
            "energy": round(eng, 1),
            "source": "video",
            "seed_id": seed["id"],
        })
    points.reverse()
    points.append({
        "frame_i": f0,
        "t_sec": round(f0 / fps, 4),
        "x": cx0,
        "y": cy0,
        "energy": float(seed.get("score") or 1) * 100,
        "source": "seed",
        "seed_id": seed["id"],
    })
    cx, cy = cx0, cy0
    for fi in range(f0, end):
        if fi not in grays or fi + 1 not in grays:
            break
        d = cv2.absdiff(grays[fi + 1], grays[fi])
        hit = _peak_in_window(d, cx, cy + 3, TRACK_WIN)
        if hit is None:
            break
        cx, cy, eng = hit
        points.append({
            "frame_i": fi + 1,
            "t_sec": round((fi + 1) / fps, 4),
            "x": int(cx),
            "y": int(cy),
            "energy": round(eng, 1),
            "source": "video",
            "seed_id": seed["id"],
        })
    uniq = {}
    for p in points:
        uniq[(p["frame_i"], p["seed_id"])] = p
    return [uniq[k] for k in sorted(uniq)]


def main() -> int:
    t0_geo = 1.6
    if os.path.isfile(ALIGN):
        t0_geo = float(json.load(open(ALIGN, encoding="utf-8")).get("geometry_t0_offset") or 1.6)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"не открыть {VIDEO}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = nframes / fps if fps else 0

    # окно атомов: birth∈[0,2.34] → t∈[t0, t0+2.4]; плюс запас ±0.5
    win0 = max(0.0, t0_geo - 0.3)
    win1 = min(dur, t0_geo + 2.5 + 0.5)
    # также полный прогон до 25с для запаса событий
    raw_focus = detect_dense(cap, fps, w, h, win0, win1)
    raw_full = detect_dense(cap, fps, w, h, 0.0, min(25.0, dur))
    cap.release()

    # приоритет окну атомов: поднять score
    boosted = []
    for score, fi, cx, cy, area, aspect in raw_focus:
        boosted.append((score * 1.35, fi, cx, cy, area, aspect))
    raw_all = boosted + list(raw_full)
    chosen = select_events(raw_all, fps, TOP_EVENTS)

    smax = max((t[0] for t in chosen), default=1.0) or 1.0
    events = []
    for i, (score, fi, cx, cy, area, aspect) in enumerate(chosen):
        half = CROP // 2
        bbox = [max(0, cx - half), max(0, cy - half), CROP, CROP]
        events.append({
            "id": f"drop_{i:04d}",
            "событие": "drop_impact",
            "video": "данные/стихии_живые/dozhd/video_live_01.mp4",
            "frame_i": int(fi),
            "t_sec": round(fi / fps, 4),
            "cx": int(cx),
            "cy": int(cy),
            "xy": [int(cx), int(cy)],
            "bbox": bbox,
            "crop_side": CROP,
            "area_contur": round(area, 1),
            "aspect": round(aspect, 2),
            "фаза": "impact",
            "score": round(float(score / smax), 4),
            "звук": "native_aac",
            "правило": "плотная геометрия: много капель/кадр, малый ROI",
        })

    # треки: топ seeds в окне атомов
    in_win = [e for e in events if win0 <= e["t_sec"] <= win1]
    seeds = sorted(in_win, key=lambda e: e["score"], reverse=True)[:40]
    all_points = []
    for seed in seeds:
        pts = track_from_seed(VIDEO, seed, fps)
        all_points.extend(pts)

    # уникальные точки трека (по frame+округлению xy)
    uniq = {}
    for p in all_points:
        key = (p["frame_i"], p["x"] // 4, p["y"] // 4)
        if key not in uniq or (p.get("energy") or 0) > (uniq[key].get("energy") or 0):
            uniq[key] = p
    points = [uniq[k] for k in sorted(uniq, key=lambda k: (k[0], k[1], k[2]))]

    # preview: кадр середины окна с точками событий
    mid_t = (win0 + win1) / 2
    mid_f = int(mid_t * fps)
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
    ok, fr = cap.read()
    cap.release()
    if ok and fr is not None:
        for e in events:
            if abs(e["t_sec"] - mid_t) < 0.35:
                cv2.circle(fr, (e["cx"], e["cy"]), 5, (0, 255, 255), 1)
        for p in points:
            if abs(p["t_sec"] - mid_t) < 0.5:
                cv2.circle(fr, (p["x"], p["y"]), 3, (0, 180, 255), -1)
        os.makedirs(os.path.dirname(OUT_PREVIEW), exist_ok=True)
        cv2.imwrite(OUT_PREVIEW, fr)

    # сохранить события (совместимость со старым именем)
    ev_report = {
        "дата": date.today().isoformat(),
        "video": os.path.relpath(VIDEO, КОРЕНЬ),
        "fps": fps,
        "режим": "dense_multi_blob",
        "окно_атомов_sec": [round(win0, 3), round(win1, 3)],
        "geometry_t0_offset": t0_geo,
        "прогон_сек": min(25.0, dur),
        "crop_side_px": CROP,
        "n_events": len(events),
        "n_raw_focus": len(raw_focus),
        "n_raw_full": len(raw_full),
        "events": events,
        "note": "плотная геометрия для xy атомов; не пейзаж",
    }
    with open(OUT_EVENTS, "w", encoding="utf-8") as f:
        json.dump(ev_report, f, ensure_ascii=False, indent=2)

    lines = [
        "# События drop_impact — плотная геометрия",
        "",
        f"> {ev_report['дата']} · N={len(events)} · окно атомов [{win0:.2f},{win1:.2f}] · t0_geo={t0_geo}",
        "",
        f"Превью: `{os.path.relpath(OUT_PREVIEW, КОРЕНЬ)}`",
        "",
        "| id | t_sec | cx,cy | area | score |",
        "|----|------:|-------|-----:|------:|",
    ]
    for e in events[:80]:
        lines.append(f"| `{e['id']}` | {e['t_sec']} | {e['cx']},{e['cy']} | {e['area_contur']} | {e['score']} |")
    if len(events) > 80:
        lines.append(f"| … | … | … | … | ещё {len(events) - 80} |")
    lines += ["", f"JSON: `{os.path.relpath(OUT_EVENTS, КОРЕНЬ)}`", ""]
    with open(OUT_EVENTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # трек-файл: все точки + мета
    primary = seeds[0] if seeds else (events[0] if events else None)
    track_report = {
        "дата": date.today().isoformat(),
        "слой": "геометрия_плотная_мультитрек",
        "seed_event": primary["id"] if primary else None,
        "n_seeds": len(seeds),
        "video": "данные/стихии_живые/dozhd/video_live_01.mp4",
        "fps": fps,
        "points": points,
        "t_impact": primary["t_sec"] if primary else None,
        "frame_impact": primary["frame_i"] if primary else None,
        "xy_impact": [primary["cx"], primary["cy"]] if primary else None,
        "approx_linear_fallback": False,
        "note": "мультитрек от плотных seeds в окне атомов",
        "preview": os.path.relpath(OUT_PREVIEW, КОРЕНЬ),
    }
    with open(OUT_TRACK, "w", encoding="utf-8") as f:
        json.dump(track_report, f, ensure_ascii=False, indent=2)
    with open(OUT_TRACK_MD, "w", encoding="utf-8") as f:
        f.write(
            f"# Трек капель — плотная геометрия\n\n"
            f"> seeds={len(seeds)} · точек={len(points)} · окно [{win0:.2f},{win1:.2f}]\n\n"
            f"Превью: `{track_report['preview']}`\n"
        )

    dense_meta = {
        "дата": date.today().isoformat(),
        "n_events": len(events),
        "n_track_points": len(points),
        "n_seeds": len(seeds),
        "окно_атомов_sec": [win0, win1],
        "geometry_t0_offset": t0_geo,
        "events_in_window": sum(1 for e in events if win0 <= e["t_sec"] <= win1),
        "preview": os.path.relpath(OUT_PREVIEW, КОРЕНЬ),
    }
    os.makedirs(os.path.dirname(OUT_DENSE), exist_ok=True)
    with open(OUT_DENSE, "w", encoding="utf-8") as f:
        json.dump(dense_meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(dense_meta, ensure_ascii=False, indent=2))
    return 0 if len(events) >= 50 else 2


if __name__ == "__main__":
    raise SystemExit(main())

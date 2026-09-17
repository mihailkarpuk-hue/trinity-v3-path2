# -*- coding: utf-8 -*-
"""Внешний синхрон: 5с video_live_01 + вспышки частиц ∝ атомам + звук сборки.

Запуск: python3 scripts/видео_синхрон_5с_частицы.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
REPORT = os.path.join(КОРЕНЬ, "отчёты", "сборка_5с_частицы_атомы.json")
MIX = os.path.join(КОРЕНЬ, "выход", "сборка_5с_частицы", "сборка_из_частиц_атомов.wav")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")

OUT_DIR = os.path.join(КОРЕНЬ, "выход", "сборка_5с_частицы")
OUT_SILENT = os.path.join(OUT_DIR, "_sync_silent.mp4")
OUT_MP4 = os.path.join(OUT_DIR, "синхрон_частицы_5с.mp4")
OUT_PREVIEW = os.path.join(OUT_DIR, "синхрон_preview_grid.jpg")

DUR = 5.0
FPS = 30.0
FLASH_SEC = 0.18  # сколько кадров светится удар


def _ff():
    return FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"


def draw_drop(frame, cx, cy, r_px, alpha, kind: str):
    """Капля + кольцо ряби поверх кадра."""
    ov = frame.copy()
    r = max(3, int(r_px))
    if kind == "detector":
        color = (210, 230, 255)  # BGR тёплый блик
        ring = (160, 210, 255)
    else:
        color = (200, 255, 220)
        ring = (140, 220, 180)
    cv2.circle(ov, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)
    # блик
    cv2.circle(ov, (cx - r // 3, cy - r // 3), max(1, r // 4), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    # рябь
    for k in (1.6, 2.4, 3.2):
        rr = int(r * k * (0.6 + 0.4 * alpha))
        cv2.circle(ov, (cx, cy), rr, ring, 1, lineType=cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def layout_xy(i: int, n: int, w: int, h: int, kind: str):
    """Детерминированная позиция: detector — ближе к центру лужи; birth — сетка сверху."""
    rng = np.random.RandomState(1000 + i * 17)
    if kind == "detector":
        cx = int(w * (0.35 + 0.3 * rng.rand()))
        cy = int(h * (0.45 + 0.35 * rng.rand()))
    else:
        # равномернее по кадру, слегка сверху (падение)
        cx = int(40 + (w - 80) * ((i * 0.618033) % 1.0))
        cy = int(60 + (h * 0.55) * ((i * 0.381966) % 1.0))
    return cx, cy


def main() -> int:
    rep = json.load(open(REPORT, encoding="utf-8"))
    particles = rep["particles"]
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit("не открыл video_live_01")
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)

    # целевой fps 30, duration 5s
    n_frames = int(DUR * FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_SILENT, fourcc, FPS, (vw, vh))

    # предрасчёт позиций
    placed = []
    for i, p in enumerate(particles):
        kind = p.get("from") or "cell_birth"
        cx, cy = layout_xy(i, len(particles), vw, vh, kind)
        r_px = 4 + float(p.get("r_mm") or 1.0) * 6.0  # ~1mm → 10px
        if kind == "detector":
            r_px *= 1.6
        placed.append({**p, "cx": cx, "cy": cy, "r_px": r_px, "kind": kind})

    preview_frames = []
    for fi in range(n_frames):
        t = fi / FPS
        # кадр из исходника на том же t
        src_fi = int(round(t * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_fi)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((vh, vw, 3), dtype=np.uint8)

        # затемнение чуть — чтобы капли читались
        frame = (frame.astype(np.float32) * 0.72).astype(np.uint8)

        active = 0
        for p in placed:
            dt = t - float(p["t_in_window"])
            if 0 <= dt < FLASH_SEC:
                alpha = float(math.exp(-dt / (FLASH_SEC * 0.45)))
                alpha = max(0.15, min(0.95, alpha))
                draw_drop(frame, p["cx"], p["cy"], p["r_px"], alpha, p["kind"])
                active += 1

        # HUD
        bar_h = 36
        cv2.rectangle(frame, (0, 0), (vw, bar_h), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"t={t:0.2f}s  particles_on={active}  total={len(placed)}  sound=сборка_из_частиц",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 230, 240),
            1,
            cv2.LINE_AA,
        )
        # timeline ticks
        y0 = vh - 28
        cv2.rectangle(frame, (0, y0), (vw, vh), (0, 0, 0), -1)
        for p in placed:
            x = int((float(p["t_in_window"]) / DUR) * (vw - 2))
            col = (180, 210, 255) if p["kind"] == "detector" else (160, 220, 180)
            cv2.line(frame, (x, y0 + 4), (x, vh - 4), col, 1)
        playhead = int((t / DUR) * (vw - 2))
        cv2.line(frame, (playhead, y0), (playhead, vh), (255, 255, 255), 2)

        writer.write(frame)
        # превью-кадры ~0.5 / 1.7 / 2.3 / 3.0 / 4.0
        if abs(t - 0.5) < 1 / FPS or abs(t - 1.67) < 1 / FPS or abs(t - 2.30) < 1 / FPS \
                or abs(t - 3.04) < 1 / FPS or abs(t - 4.04) < 1 / FPS:
            preview_frames.append(frame.copy())

    writer.release()
    cap.release()

    # mux audio
    cmd = [
        _ff(), "-y",
        "-i", OUT_SILENT,
        "-i", MIX,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        OUT_MP4,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg mux fail:\n{r.stderr[-800:]}")

    # grid preview
    if preview_frames:
        thumbs = [cv2.resize(f, (480, 360)) for f in preview_frames[:5]]
        while len(thumbs) < 5:
            thumbs.append(np.zeros_like(thumbs[0]))
        row = np.hstack(thumbs)
        cv2.imwrite(OUT_PREVIEW, row)

    try:
        os.remove(OUT_SILENT)
    except OSError:
        pass

    print(json.dumps({
        "ok": True,
        "mp4": OUT_MP4,
        "preview": OUT_PREVIEW,
        "n_particles": len(placed),
        "dur": DUR,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

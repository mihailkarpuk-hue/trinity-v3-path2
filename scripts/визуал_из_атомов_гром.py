# -*- coding: utf-8 -*-
"""Визуал грома v2: вспышка-канал + раскат-поле. БЕЗ кружков (E).

Запуск: python3 scripts/визуал_из_атомов_гром.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATOMS = os.path.join(КОРЕНЬ, "выход", "атомы_полные_гром", "атомы_звук_образ.json")
SOUND = os.path.join(КОРЕНЬ, "выход", "причина_гром", "калибр_оси", "сборка_104_чистая.wav")
FALLBACK = os.path.join(КОРЕНЬ, "выход", "причина_гром", "калибр_оси", "сборка_без_цикла.wav")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_гром", "визуал")
OUT_SILENT = os.path.join(OUT_DIR, "_silent.mp4")
OUT_MP4 = os.path.join(OUT_DIR, "гром_из_атомов.mp4")
OUT_PREVIEW = os.path.join(OUT_DIR, "preview.jpg")

W, H = 1280, 720
FPS = 30.0


def _ff():
    return FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"


def bolt_polyline(rng, fx, fy, segs=10):
    pts = [(fx, fy)]
    x, y = fx, fy
    for _ in range(segs):
        x += int(rng.integers(-55, 55))
        y += int(rng.integers(28, 70))
        pts.append((int(np.clip(x, 8, W - 8)), int(np.clip(y, 8, H - 8))))
    return pts


def draw_bolt(frame, pts, bright: float):
    if bright <= 0.02:
        return
    col = (int(200 * bright), int(220 * bright), int(255 * bright))
    col_soft = (int(80 * bright), int(100 * bright), int(140 * bright))
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], col_soft, 5, cv2.LINE_AA)
        cv2.line(frame, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)
    # ветка
    if len(pts) > 4:
        mid = pts[len(pts) // 2]
        tip = (mid[0] + 40, mid[1] + 50)
        cv2.line(frame, mid, tip, col, 1, cv2.LINE_AA)


def draw_rumble_field(frame, cx, cy, age, amp, size, rng):
    """Раскат = дрожащее поле / полосы давления — не кольца."""
    if age < 0 or age > 2.0:
        return
    fade = math.exp(-age * 1.1)
    a = fade * min(1.0, 0.25 + amp)
    # горизонтальные волны давления
    span = int(60 + size * 180 + age * 120)
    n_lines = 3 + int(amp * 4)
    for k in range(n_lines):
        yy = int(cy + (k - n_lines / 2) * (10 + size * 8) + 6 * math.sin(age * 28 + k))
        if yy < 4 or yy >= H - 4:
            continue
        bri = a * (1.0 - abs(k - n_lines / 2) / (n_lines / 2 + 1))
        col = (int(70 * bri), int(95 * bri), int(125 * bri))
        x0 = max(4, cx - span)
        x1 = min(W - 4, cx + span)
        # ломаная, не круг
        pts = []
        for x in range(x0, x1, 18):
            jitter = int(4 * math.sin(0.08 * x + age * 35 + k) * (1 + amp))
            pts.append((x, int(np.clip(yy + jitter, 2, H - 2))))
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)
    # вертикальная «тряска» столба воздуха (намёк ударной волны)
    if age < 0.6:
        hspan = int(30 + age * 80)
        for dx in (-hspan // 2, 0, hspan // 2):
            x = int(np.clip(cx + dx, 4, W - 4))
            col = (int(50 * a), int(70 * a), int(95 * a))
            y0 = max(4, cy - hspan)
            y1 = min(H - 4, cy + hspan)
            cv2.line(frame, (x, y0), (x, y1), col, 1, cv2.LINE_AA)


def main() -> int:
    pkg = json.load(open(ATOMS, encoding="utf-8"))
    atoms = pkg["atoms"]
    births = [float(a.get("birth") or 0) for a in atoms]
    dur = max(2.5, max(births) + 0.8)
    n_frames = int(dur * FPS)

    step = max(1, len(atoms) // 700)
    sample = atoms[::step]
    rng = np.random.default_rng(7)

    t_flash_vis = 0.12
    bolt = bolt_polyline(rng, int(W * 0.52), int(H * 0.08))

    impacts = []
    for a in sample:
        birth = float(a.get("birth") or 0)
        core = (a.get("звук_ядро") or {}).get("ядро") or {}
        amp = float(core.get("amp") or 0.1)
        size = float(core.get("size") or 0.1)
        # раскат смещён вниз от канала вспышки (не HUD-точки по кадру)
        cx = int(np.clip(bolt[min(5, len(bolt) - 1)][0] + rng.normal(0, 90), 80, W - 80))
        cy = int(np.clip(bolt[-1][1] + rng.uniform(20, 180), 100, H - 60))
        impacts.append({"t": birth, "cx": cx, "cy": cy, "amp": amp, "size": size})

    os.makedirs(OUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(OUT_SILENT, fourcc, FPS, (W, H))

    for fi in range(n_frames):
        t = fi / FPS
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            v = int(6 + 14 * (y / H))
            frame[y, :] = (v, v, int(v * 1.15))

        age_f = t - t_flash_vis
        if -0.05 <= age_f <= 0.45:
            bright = math.exp(-max(0, age_f) * 7.0) if age_f >= 0 else 0.12
            # вспышка кадра мягко (не заливка пейзажа)
            if age_f >= 0 and age_f < 0.12:
                frame = np.clip(frame.astype(np.int16) + int(90 * bright), 0, 255).astype(np.uint8)
            draw_bolt(frame, bolt, min(1.0, bright * 1.35))

        for im in impacts:
            draw_rumble_field(frame, im["cx"], im["cy"], t - im["t"], im["amp"], im["size"], rng)

        wr.write(frame)
        if fi == int(0.15 * FPS):
            cv2.imwrite(OUT_PREVIEW, frame)
        if fi == int(n_frames * 0.45):
            cv2.imwrite(OUT_PREVIEW.replace(".jpg", "_mid.jpg"), frame)

    wr.release()

    snd = SOUND if os.path.isfile(SOUND) else FALLBACK
    if not os.path.isfile(snd):
        snd = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "grom_real.wav")
    subprocess.run([
        _ff(), "-y", "-i", OUT_SILENT, "-i", snd,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", OUT_MP4,
    ], capture_output=True)
    print(json.dumps({
        "ok": os.path.isfile(OUT_MP4),
        "mp4": os.path.relpath(OUT_MP4, КОРЕНЬ),
        "версия": "v2_без_кружков",
        "n_impacts": len(impacts),
        "дата": date.today().isoformat(),
    }, ensure_ascii=False, indent=2))
    return 0 if os.path.isfile(OUT_MP4) else 2


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Визуал из атомов: одна траектория на каплю (верх → удар), без двух слоёв.

Плотность: тот же путь повторяется с периодом (та же линия, тот же удар).

Запуск: python3 scripts/визуал_из_атомов_дождь.py
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
ATOMS = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "атомы_звук_образ.json")
SOUND = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "калибр_оси", "сборка_без_цикла.wav")
FALLBACK_SOUND = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "dozhd_real.wav")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")

OUT_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "визуал")
OUT_SILENT = os.path.join(OUT_DIR, "_silent.mp4")
OUT_MP4 = os.path.join(OUT_DIR, "дождь_из_атомов.mp4")
OUT_PREVIEW = os.path.join(OUT_DIR, "preview_grid.jpg")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "E_визуал_из_атомов.html")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "визуал_из_атомов_дождь.md")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "визуал_из_атомов_дождь.json")

W, H = 1280, 720
FPS = 30.0
DUR = 2.4


def _ff():
    return FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"


def draw_streak(frame, x0, y0, x1, y1, amp: float, thick: int):
    n = 5
    for i in range(n):
        t0 = i / n
        t1 = (i + 1) / n
        xa = int(x0 + (x1 - x0) * t0)
        ya = int(y0 + (y1 - y0) * t0)
        xb = int(x0 + (x1 - x0) * t1)
        yb = int(y0 + (y1 - y0) * t1)
        bri = 0.2 + 0.8 * t1
        a = float(np.clip(amp * bri, 0.05, 1.0))
        col = (int(150 * a), int(185 * a), int(230 * a))
        cv2.line(frame, (xa, ya), (xb, yb), col, max(1, thick), lineType=cv2.LINE_AA)


def draw_impact(frame, cx, cy, r_mm: float, age: float, amp: float):
    if age < 0 or age > 1:
        return
    fade = math.exp(-age * 4.0)
    r0 = max(1, int(1.2 + r_mm * 0.8))
    a = fade * min(1.0, 0.4 + amp)
    col = (int(200 * a), int(220 * a), int(255 * a))
    cv2.circle(frame, (cx, cy), r0, col, 1, lineType=cv2.LINE_AA)
    if age < 0.35:
        for ang in (-50, -20, 20, 50):
            rad = math.radians(ang)
            L = int((6 + r_mm * 4) * (1 - age / 0.35) * fade)
            x2 = int(cx + L * math.sin(rad))
            y2 = int(cy - L * math.cos(rad))
            cv2.line(frame, (cx, cy), (x2, y2), col, 1, lineType=cv2.LINE_AA)
    for k, scale in enumerate((2.5, 4.0, 6.0)):
        rr = int((3 + r_mm * 2) * scale * (0.4 + age))
        aa = fade * (0.55 - k * 0.15)
        if aa < 0.05:
            continue
        c2 = (int(120 * aa), int(160 * aa), int(200 * aa))
        cv2.circle(frame, (cx, cy), rr, c2, 1, lineType=cv2.LINE_AA)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    atoms = json.load(open(ATOMS, encoding="utf-8"))["atoms"]
    n_frames = int(DUR * FPS)
    ground_y = int(H * 0.78)

    prepared = []
    for i, a in enumerate(atoms):
        yadro = a["звук_ядро"]["ядро"]
        zak = a["образ_причины"]["закон"]
        birth = float(a.get("birth") or 0.0)
        amp = float(yadro.get("amp") or 0.3)
        r_mm = float(zak.get("r_mm") or 1.0)
        v_ms = float(zak.get("v_ms") or 6.0)
        rng = np.random.default_rng(10007 + i)
        geo = a["образ_причины"].get("геометрия") or {}

        # конец пути = удар
        if geo.get("xy") and geo["xy"][0] is not None:
            ex = int(np.clip(float(geo["xy"][0]) / 1440 * W, 8, W - 8))
            ey = int(np.clip(float(geo["xy"][1]) / 1080 * H, ground_y - 50, H - 8))
        else:
            px = yadro.get("pos_x")
            ex = int(np.clip((float(px) + 3) / 6 * W, 8, W - 8)) if px is not None else int(rng.uniform(12, W - 12))
            ey = int(np.clip(ground_y + rng.integers(-40, 50), int(H * 0.58), H - 6))

        # начало пути — ТА ЖЕ прямая: сверху над точкой удара с лёгким ветром
        wind = float(rng.uniform(-0.06, 0.14))
        sy = float(rng.uniform(2, H * 0.12))
        fall = max(30.0, ey - sy)
        sx = float(ex) - wind * fall  # прямая (sx,sy)→(ex,ey)

        flight_s = float(np.clip(fall / (300 + v_ms * 35), 0.28, 0.55))
        impact_s = 0.09 + 0.02 * r_mm
        gap_s = 0.05
        period = flight_s + impact_s + gap_s

        prepared.append({
            "birth": birth,
            "amp": amp,
            "r_mm": r_mm,
            "sx": sx, "sy": sy,
            "ex": float(ex), "ey": float(ey),
            "flight_s": flight_s,
            "impact_s": impact_s,
            "period": period,
            "thick": 1 if r_mm < 1.35 else 2,
            "trail": min(85.0, fall * 0.25),
        })

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_SILENT, fourcc, FPS, (W, H))
    previews = []

    for fi in range(n_frames):
        t = fi / FPS
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (12, 11, 10)
        for y in range(ground_y, H):
            k = (y - ground_y) / max(1, H - ground_y)
            frame[y, :] = (int(12 + 18 * k), int(14 + 22 * k), int(16 + 28 * k))
        cv2.line(frame, (0, ground_y), (W, ground_y), (40, 45, 50), 1, lineType=cv2.LINE_AA)

        n_flight = n_impact = 0
        for p in prepared:
            # одна фаза на период; birth задаёт сдвиг, чтобы удар совпал с birth в первом цикле
            # tau=0 — старт сверху; tau=flight_s — удар
            t0 = p["birth"] - p["flight_s"]  # момент старта цикла, где удар = birth
            tau = (t - t0) % p["period"]
            if tau < p["flight_s"]:
                u = tau / p["flight_s"]  # 0 сверху → 1 в точке удара
                hx = p["sx"] + (p["ex"] - p["sx"]) * u
                hy = p["sy"] + (p["ey"] - p["sy"]) * u
                # хвост назад по той же прямой
                u_tail = max(0.0, u - p["trail"] / max(p["ey"] - p["sy"], 1.0))
                tx = p["sx"] + (p["ex"] - p["sx"]) * u_tail
                ty = p["sy"] + (p["ey"] - p["sy"]) * u_tail
                draw_streak(frame, int(tx), int(ty), int(hx), int(hy), p["amp"], p["thick"])
                n_flight += 1
            elif tau < p["flight_s"] + p["impact_s"]:
                age = (tau - p["flight_s"]) / p["impact_s"]
                # удар строго в (ex, ey) — конец той же траектории
                draw_impact(frame, int(p["ex"]), int(p["ey"]), p["r_mm"], age, p["amp"])
                n_impact += 1

        cv2.rectangle(frame, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"one path/atom  t={t:.2f}s  flight={n_flight} impact={n_impact}",
            (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (190, 200, 210),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
        if fi in (0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1):
            previews.append(frame.copy())

    writer.release()

    snd = SOUND if os.path.isfile(SOUND) else FALLBACK_SOUND
    cmd = [
        _ff(), "-y", "-i", OUT_SILENT, "-i", snd,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-movflags", "+faststart", OUT_MP4,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-800:])
    try:
        os.remove(OUT_SILENT)
    except OSError:
        pass

    if previews:
        cv2.imwrite(OUT_PREVIEW, np.hstack([cv2.resize(f, (256, 144)) for f in previews]))

    report = {
        "дата": date.today().isoformat(),
        "правило": "одна прямая (sx,sy)→(ex,ey) на атом; удар только в конце пути; плотность = повтор периода",
        "n_atoms": len(atoms),
        "mp4": os.path.relpath(OUT_MP4, КОРЕНЬ),
        "sound": os.path.relpath(snd, КОРЕНЬ),
        "preview": os.path.relpath(OUT_PREVIEW, КОРЕНЬ),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Визуал v3 — одна траектория\n\n" + report["правило"] + f"\n\n`{report['mp4']}`\n")

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — визуал v3 одна траектория</title>
<style>
body{{margin:0;background:#0c0c0c;color:#e8e8e8;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:24px 16px 48px}}
h1{{font:600 22px/1.2 Georgia,serif;margin:0 0 8px}}
.meta{{opacity:.75;margin:0 0 16px}}
video,img{{width:100%;background:#111}}
.card{{margin:0 0 14px;padding:14px;border:1px solid #2a2a2a;background:#161616}}
.q{{margin-top:18px;padding:14px;border-left:3px solid #6a8;background:#121812}}
</style></head><body><main>
<h1>Дождь из атомов v3</h1>
<p class="meta">Верх и низ одной линии: капля летит в свою точку удара. Повтор периода = плотность.</p>
<div class="card"><video controls autoplay loop playsinline src="/{report['mp4']}"></video></div>
<div class="card"><img src="/{report['preview']}" alt="preview"/></div>
<div class="q"><strong>E:</strong> траектории совпали? ок / почти / мимо</div>
</main></body></html>
"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps({"ok": True, "mp4": OUT_MP4, "html": OUT_HTML}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Визуал огня v4 — база СНИЗУ, жар снизу→вверх, цвет как у живого.

E: v3 горел «сверху вниз» (якорь mid + белый кончик).
v4: якорь низ холста; ядро бело-жёлтое у базы; кончик тёмно-красный (не white).

Запуск: python3 scripts/визуал_из_атомов_огонь.py
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
ATOMS = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "атомы_звук_образ.json")
SOUND = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "калибр_оси", "сборка_чистая.wav")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "визуал")
OUT_SILENT = os.path.join(OUT_DIR, "_silent.mp4")
OUT_MP4 = os.path.join(OUT_DIR, "огонь_из_атомов.mp4")
OUT_PREVIEW = os.path.join(OUT_DIR, "preview.jpg")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "E_визуал_из_атомов.html")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "визуал_из_атомов_огонь.md")

W, H = 1280, 720
FPS = 30.0
# живой кадр ~480x640 — маппим в центр холста
SRC_W, SRC_H = 480.0, 640.0

# канон с живого live_01 (BGR): низ — бело-жёлтое ядро, середина оранж, верх — тёмный красный
COL_CORE = (220, 250, 255)   # низ / ядро
COL_HOT = (60, 190, 255)     # низ-середина
COL_MID = (25, 120, 245)     # тело
COL_EDGE = (8, 35, 160)      # кромка
COL_TIP = (4, 18, 90)        # кончик (НЕ белый)
COL_WISP = (3, 12, 55)       # струйки дыма/жара наверху


def _ff():
    return FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"


def add_blob(buf, cx, cy, rx, ry, bgr, strength):
    if strength < 0.015 or rx < 0.8 or ry < 0.8:
        return
    x0, x1 = max(0, int(cx - rx * 2.4)), min(W, int(cx + rx * 2.4) + 1)
    y0, y1 = max(0, int(cy - ry * 2.4)), min(H, int(cy + ry * 2.4) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys = np.arange(y0, y1, dtype=np.float64)[:, None]
    xs = np.arange(x0, x1, dtype=np.float64)[None, :]
    d2 = ((xs - cx) / max(rx, 1e-6)) ** 2 + ((ys - cy) / max(ry, 1e-6)) ** 2
    mask = np.exp(-d2 * 1.55)
    mask[d2 > 4.5] = 0.0
    for c, val in enumerate(bgr):
        buf[y0:y1, x0:x1, c] += mask * float(val) * strength


def lerp_bgr(a, b, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat_color(frac_up: float, edge: float = 0.0):
    """frac_up 0=низ/база (жар), 1=кончик (холоднее). edge 0=ядро, 1=кромка.

    Живое: снизу яркое ядро, сверху тёмные языки — НЕ белый кончик.
    """
    if frac_up < 0.22:
        base = lerp_bgr(COL_CORE, COL_HOT, frac_up / 0.22)
    elif frac_up < 0.55:
        base = lerp_bgr(COL_HOT, COL_MID, (frac_up - 0.22) / 0.33)
    elif frac_up < 0.82:
        base = lerp_bgr(COL_MID, COL_EDGE, (frac_up - 0.55) / 0.27)
    else:
        base = lerp_bgr(COL_EDGE, COL_TIP, (frac_up - 0.82) / 0.18)
    return lerp_bgr(base, COL_EDGE, edge * 0.72)


def map_xy(xy, fallback_x, fallback_y, *, y_from_geo: bool = False):
    """X из геометрии клипа; Y по умолчанию якорь низа (огонь растёт снизу вверх)."""
    if not xy or len(xy) < 2:
        return fallback_x, fallback_y
    scale = min(W / SRC_W, H / SRC_H) * 1.25
    ox = (W - SRC_W * scale) * 0.5
    oy = (H - SRC_H * scale) * 0.55  # ниже, ближе к полу кадра
    x = ox + float(xy[0]) * scale
    if y_from_geo:
        y = oy + float(xy[1]) * scale
    else:
        y = fallback_y
    return float(np.clip(x, 60, W - 60)), float(np.clip(y, 80, H - 20))


def tongue_path(rng, x0, y0, height, lean, phase, t, segs=14):
    """Ломаная языка вверх с турбулентностью (не прямая свечка)."""
    pts = []
    x, y = x0, y0
    for i in range(segs + 1):
        frac = i / segs
        # сужение вверх + дрожь сильнее кверху
        turb = (10 + 58 * frac) * math.sin(t * 17 + phase + i * 0.55)
        turb += (8 + 40 * frac) * math.sin(t * 29 + phase * 1.7 + i * 1.1)
        turb += float(rng.normal(0, 2.0 + 7.0 * frac))
        x = x0 + lean * frac * height * 0.22 + turb
        y = y0 - height * (frac ** 0.92)
        pts.append((x, y, frac))
    return pts


def draw_tongue(buf, pts, amp, size, life, rng):
    if life < 0.04 or len(pts) < 3:
        return
    for i in range(1, len(pts)):
        x0, y0, f0 = pts[i - 1]
        x1, y1, f1 = pts[i]
        frac = 0.5 * (f0 + f1)
        # ширина: низ широкий, верх тонкий (конус), + асимметрия
        width = (55 + size * 140 + amp * 70) * (1.35 - 1.1 * frac) * life
        width = max(4.0, width)
        # тело
        col = heat_color(frac, edge=0.1)
        add_blob(buf, x1, y1, width * 0.55, width * 0.85, col, (0.45 + 0.4 * amp) * life)
        # кромка темнее по бокам
        edge_col = heat_color(frac, edge=0.85)
        add_blob(buf, x1 - width * 0.35, y1, width * 0.28, width * 0.5, edge_col, 0.22 * life)
        add_blob(buf, x1 + width * 0.35, y1, width * 0.28, width * 0.5, edge_col, 0.22 * life)
        # ядро только у базы (низ) — не на кончике
        if frac < 0.32:
            add_blob(
                buf, x1, y1,
                width * 0.28, width * 0.40,
                COL_CORE,
                (0.45 + 0.5 * amp) * life * (1.0 - frac * 1.4),
            )
    # струйки на кончике — тёмные, вверх (не белые)
    tip_x, tip_y, _ = pts[-1]
    for k in range(3 + int(amp * 4)):
        wx = tip_x + float(rng.normal(0, 10 + size * 8))
        wy = tip_y - float(rng.uniform(5, 28 + size * 20))
        add_blob(buf, wx, wy, 2.2, 6.0, COL_WISP, 0.28 * life * amp)


def draw_spark(buf, x0, y0, age, amp, v_up, phase):
    if age < 0 or age > 0.55:
        return
    fade = math.exp(-age * 5.2)
    y = y0 - (120 + 320 * v_up) * age - amp * 60 * age
    x = x0 + 28 * math.sin(age * 48 + phase) + 10 * math.sin(age * 90)
    bri = fade * min(1.0, 0.45 + amp)
    add_blob(buf, x, y, 2.0, 3.2, COL_CORE, bri)
    add_blob(buf, x, y - 5, 1.2, 2.2, COL_HOT, bri * 0.7)


def main() -> int:
    pkg = json.load(open(ATOMS, encoding="utf-8"))
    atoms = pkg["atoms"]
    births = [float(a.get("birth") or 0) for a in atoms]
    dur = max(2.8, max(births) + 0.9)
    n_frames = int(dur * FPS)
    rng = np.random.default_rng(21)

    # якорь: X из геометрии клипа, Y жёстко низ холста (рост снизу→вверх)
    xs = []
    for a in atoms[:: max(1, len(atoms) // 800)]:
        geo = (a.get("образ_причины") or {}).get("геометрия") or {}
        xy = geo.get("xy")
        if xy and xy[0] is not None:
            mx, _ = map_xy(xy, W * 0.5, H * 0.88)
            xs.append(mx)
    base_x = float(np.median(xs)) if xs else W * 0.5
    base_y = H * 0.88  # НЕ медиана geo.y (~mid кадра) — иначе «горит сверху»

    # языки: сильные атомы вокруг низа
    scored = []
    for i, a in enumerate(atoms):
        core = (a.get("звук_ядро") or {}).get("ядро") or {}
        amp = float(core.get("amp") or 0.1)
        size = float(core.get("size") or 0.1)
        scored.append((amp * 0.65 + size * 0.35, i, a))
    scored.sort(reverse=True)
    tongues = []
    for rank, (_, i, a) in enumerate(scored[: 42]):
        core = (a.get("звук_ядро") or {}).get("ядро") or {}
        zak = (a.get("образ_причины") or {}).get("закон") or {}
        geo = (a.get("образ_причины") or {}).get("геометрия") or {}
        amp = float(core.get("amp") or 0.1)
        size = float(core.get("size") or 0.1)
        v_up = float(zak.get("v_up") or 0.5)
        birth = float(a.get("birth") or 0)
        mx, my = map_xy(geo.get("xy"), base_x, base_y)
        mx = base_x * 0.55 + mx * 0.45 + float(rng.normal(0, 32))
        # Y только у базы + мелкий разброс вниз/вверх у корня
        my = base_y + float(rng.normal(0, 10))
        lean = float(rng.uniform(-0.85, 0.85))
        height = 280 + size * 480 + amp * 240 + max(0, 18 - rank) * 12
        tongues.append({
            "birth": birth,
            "x": mx,
            "y": my,
            "amp": amp,
            "size": size,
            "v_up": max(0.55, v_up),
            "lean": lean,
            "height": height,
            "phase": float(i * 0.21 + rank * 0.37),
            "persist": True,
        })

    # искры треска — плотнее, из top amp
    sparks = []
    for _, i, a in scored[:180]:
        core = (a.get("звук_ядро") or {}).get("ядро") or {}
        zak = (a.get("образ_причины") or {}).get("закон") or {}
        amp = float(core.get("amp") or 0.1)
        if amp < 0.08:
            continue
        sparks.append({
            "t": float(a.get("birth") or 0),
            "x": base_x + float(rng.normal(0, 55)),
            "y": base_y - float(rng.uniform(0, 40)),
            "amp": amp,
            "v_up": float(zak.get("v_up") or 0.5),
            "phase": float(i * 0.13),
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    wr = cv2.VideoWriter(OUT_SILENT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    for fi in range(n_frames):
        t = fi / FPS
        frame = np.zeros((H, W, 3), dtype=np.float64)
        # чёрный фон как в живом (без «пола-чаши»)
        # лёгкая аура жара под очагом
        add_blob(frame, base_x, base_y + 18, 160, 42, (6, 18, 40), 0.35)

        # пульс очага от плотности nearby births
        pulse = 0.85 + 0.15 * math.sin(t * 11.0)

        for tg in tongues:
            # язык всегда виден, но вспыхивает сильнее около birth
            age = t - tg["birth"]
            flash = 1.0
            if 0 <= age <= 0.35:
                flash = 1.0 + 0.55 * math.exp(-age * 6)
            elif age < 0:
                flash = 0.75
            life = pulse * (0.55 + 0.45 * tg["amp"]) * flash
            h = tg["height"] * (0.85 + 0.2 * tg["v_up"]) * (0.9 + 0.15 * math.sin(t * 9 + tg["phase"]))
            pts = tongue_path(rng, tg["x"], tg["y"], h, tg["lean"], tg["phase"], t)
            draw_tongue(frame, pts, tg["amp"], tg["size"], life, rng)

        for sp in sparks:
            draw_spark(frame, sp["x"], sp["y"], t - sp["t"], sp["amp"], sp["v_up"], sp["phase"])

        out = np.clip(frame, 0, 255).astype(np.uint8)
        # лёгкое bloom как у живого пересвета ядра
        glow = cv2.GaussianBlur(out, (0, 0), 4.0)
        out = cv2.addWeighted(out, 0.72, glow, 0.28, 0)
        wr.write(out)
        if fi == int(0.45 * FPS):
            cv2.imwrite(OUT_PREVIEW, out)
        if fi == int(n_frames * 0.5):
            cv2.imwrite(OUT_PREVIEW.replace(".jpg", "_mid.jpg"), out)

    wr.release()

    snd = SOUND if os.path.isfile(SOUND) else os.path.join(
        КОРЕНЬ, "выход", "причина_огонь", "live_sync", "ogon_live_01_etalon.wav"
    )
    # громче в mux
    subprocess.run([
        _ff(), "-y", "-i", OUT_SILENT, "-i", snd,
        "-filter:a", "volume=2.2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", OUT_MP4,
    ], capture_output=True)

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E глаз — огонь v4</title>
<style>body{{font-family:system-ui;background:#100c08;color:#f2e6d8;max-width:900px;margin:2rem auto;padding:0 1rem}}
video{{width:100%;background:#000}}.meta{{opacity:.85}}</style></head><body>
<h1>E глаз — огонь v4</h1>
<p class="meta">{date.today().isoformat()} · база снизу · жар↑ · кончик тёмный</p>
<video controls src="визуал/огонь_из_атомов.mp4"></video>
<p><a href="/выход/причина_огонь/E_сравнение_живое_vs_атомы.html" style="color:#fc6">сравнение с живым</a></p>
</body></html>""")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(
            f"# Визуал огонь v4\n\n"
            f"> {date.today().isoformat()} · эталон = video_live_01\n\n"
            f"vs v3: якорь Y=низ (не mid geo), цвет низ→яркий / верх→тёмный, без белого tip.\n"
        )

    print(json.dumps({
        "ok": os.path.isfile(OUT_MP4),
        "версия": "v4_снизу_вверх",
        "base_y": base_y,
        "n_tongues": len(tongues),
        "n_sparks": len(sparks),
        "mp4": os.path.relpath(OUT_MP4, КОРЕНЬ),
        "дата": date.today().isoformat(),
    }, ensure_ascii=False, indent=2))
    return 0 if os.path.isfile(OUT_MP4) else 2


if __name__ == "__main__":
    raise SystemExit(main())

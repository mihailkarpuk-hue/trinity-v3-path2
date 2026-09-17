# -*- coding: utf-8 -*-
"""Материя образа — разные примитивы для разных стихий.

Канон: ворота/КАК_собираем_атомы/ПРАВИЛО_материя_образа.md
Запрет: один Gaussian-blob на все стихии с разной только траекторией.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# ── палитры (развести вещества) ─────────────────────────────────────
# ветер — холодный воздух, почти бесцветный
WIND_FIBER = (195, 185, 165)      # BGR: пыльно-циановый
WIND_CORE = (220, 210, 195)
WIND_FLECK = (180, 170, 155)

# водопад — плотная белая вода + холодный сине-серый
FALL_RIBBON = (210, 175, 120)     # холоднее, больше синего в B
FALL_CORE = (245, 235, 220)
FALL_SPRAY = (255, 245, 235)
FALL_MIST = (190, 160, 130)

# река — бирюзово-зеленоватая масса русла
RIVER_BODY = (160, 140, 70)       # зеленовато-бирюзовый
RIVER_SHEET = (190, 165, 95)
RIVER_SPEC = (230, 220, 200)      # блик
RIVER_FOAM = (235, 230, 220)


def add_blob(buf, cx, cy, rx, ry, bgr, strength, *, hard=False):
    """soft Gaussian или harder disk (брызги)."""
    h, w = buf.shape[:2]
    if strength < 0.01 or rx < 0.4 or ry < 0.4:
        return
    x0, x1 = max(0, int(cx - rx * 2.2)), min(w, int(cx + rx * 2.2) + 1)
    y0, y1 = max(0, int(cy - ry * 2.2)), min(h, int(cy + ry * 2.2) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys = np.arange(y0, y1, dtype=np.float64)[:, None]
    xs = np.arange(x0, x1, dtype=np.float64)[None, :]
    d2 = ((xs - cx) / max(rx, 1e-6)) ** 2 + ((ys - cy) / max(ry, 1e-6)) ** 2
    if hard:
        mask = (d2 <= 1.0).astype(np.float64) * strength
        mask *= np.clip(1.15 - d2, 0, 1)  # мягкий край диска
    else:
        mask = np.exp(-d2 * (2.2 if rx < 3 else 1.5))
        mask[d2 > 4.0] = 0.0
        mask *= strength
    for c, val in enumerate(bgr):
        buf[y0:y1, x0:x1, c] += mask * float(val)


def stroke_line(buf, x0, y0, x1, y1, bgr, thick, strength):
    """Тонкая линия (рябь / волокно) — не blob."""
    h, w = buf.shape[:2]
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 1.5))
    for i in range(n):
        t = i / (n - 1)
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        add_blob(buf, x, y, thick * 0.55, thick * 0.35, bgr, strength * (0.7 + 0.3 * (1 - abs(t - 0.5) * 2)), hard=False)


# ── ВЕТЕР: рваные тонкие волокна + flecks ───────────────────────────

def render_wind(frame, t, streams, flecks, W, H, sx=1.0):
    """Материя: thin broken fibers (не толстые blob-ленты)."""
    frame[:, :] = (14, 12, 11)
    add_blob(frame, W * 0.5, H * 0.4, W * 0.5, H * 0.35, (22, 20, 18), 0.25)

    for st in streams:
        age = t - st["birth"]
        if age < -0.05:
            continue
        life = 0.5 + 0.5 * st["amp"]
        span = st["len"]
        x_head = (age * 155 * abs(st["v_x"]) + st["seed"] * W) % (W + span)
        if sx < 0:
            x_head = W - x_head
        # рваные сегменты: пропускаем дыры
        segs = 22
        for i in range(segs):
            if (hash((st["seed"], i)) % 7) < 2:  # дырки в волокне
                continue
            frac = i / segs
            x = x_head - sx * span * frac
            y = st["y0"] + math.sin(x * 0.018 + st["phase"] + t * 4.0) * st["wave"]
            y += math.sin(x * 0.05 + st["phase"] * 2) * st["wave"] * 0.25
            fade = (1.0 - frac) * life
            # очень тонкое волокно: rx>>ry но ry мал
            tw = max(0.7, st["thick"] * 0.22 * (0.5 + 0.5 * (1 - frac)))
            col = WIND_CORE if frac < 0.15 else WIND_FIBER
            add_blob(frame, x, y, tw * 2.8, tw * 0.45, col, 0.18 * fade)
            # короткий штрих вдоль потока
            stroke_line(
                frame, x, y, x - sx * 9, y + math.sin(st["phase"] + i) * 2,
                WIND_FIBER, 0.6, 0.12 * fade,
            )

    for d in flecks:
        age = t - d["t0"]
        if age < 0 or age > 1.2:
            continue
        x = d["x0"] + d["v"] * age
        y = d["y"] + math.sin(age * 11 + d["phase"]) * 18
        fade = math.exp(-age * 1.6) * (0.35 + d["amp"])
        # fleck = почти точка, hard
        add_blob(frame, x, y, 1.1, 1.1, WIND_FLECK, 0.55 * fade, hard=True)


# ── ВОДОПАД: плотные вертикальные ленты + крупные брызги ────────────

def render_waterfall(frame, t, ribbons, sprays, W, H):
    """Материя: continuous vertical ribbons + hard spray disks."""
    frame[:, :] = (12, 10, 9)
    add_blob(frame, W * 0.5, H * 0.12, W * 0.4, H * 0.1, (24, 20, 16), 0.35)

    for st in ribbons:
        age = t - st["birth"]
        if age < -0.05:
            continue
        y_head = (-50 + age * 240 * st["v_y"] + st["seed"] * 90) % (H + st["len"])
        segs = 20
        for i in range(segs):
            frac = i / segs
            y = y_head - st["len"] * frac
            sway = math.sin(y * 0.012 + st["phase"] + t * 2.0) * (6 + st["amp"] * 14)
            x = st["x0"] + sway
            fade = (1.0 - frac) * (0.55 + 0.45 * st["amp"])
            # плотная лента: вертикально вытянутый blob, толще ветра
            tw = st["thick"] * (0.7 + 0.5 * (1 - frac))
            col = FALL_CORE if frac < 0.12 else FALL_RIBBON
            add_blob(frame, x, y, tw * 0.7, tw * 2.1, col, 0.28 * fade)
            # второй слой — «лист» воды шире
            add_blob(frame, x, y, tw * 1.3, tw * 1.5, FALL_RIBBON, 0.12 * fade)

    # зона удара — плотный туман
    add_blob(frame, W * 0.5, H * 0.82, W * 0.35, H * 0.12, FALL_MIST, 0.2)

    for s in sprays:
        age = t - s["t0"]
        if age < 0 or age > 1.4:
            continue
        x = s["x"] + math.sin(age * 6 + s["phase"]) * 25
        y = s["y0"] + age * 55 + math.cos(s["phase"]) * 10
        fade = math.exp(-age * 1.3) * s["amp"]
        r = 2.5 + s["amp"] * 8  # крупные брызги
        add_blob(frame, x, y, r, r, FALL_SPRAY, 0.5 * fade, hard=True)
        if s["amp"] > 0.2:
            add_blob(frame, x + 4, y - 3, r * 0.5, r * 0.5, FALL_CORE, 0.25 * fade, hard=True)


# ── РЕКА: параллельные струи течения + пена (НЕ эллипс-blob) ─────────

def render_river(frame, t, currents, foam, rocks, W, H, sx=1.0):
    """Материя: плотное русло + яркие параллельные линии тока + пена.
    Запрет: эллипс-линза, тёмные круги-диски, soft-blob на весь кадр, сильный glow.
    rocks игнорируются (совместимость API).
    """
    _ = rocks
    # весь кадр — вода (не полоса-«линза» по центру)
    frame[:, :] = (18, 15, 10)
    y_top, y_bot = 0, H
    base = np.array([88.0, 72.0, 34.0])
    ys = np.arange(H, dtype=np.float64)
    u = ys / max(1.0, H - 1)
    # лёгкая глубина, без сильного mid-glow (он даёт ложный овал)
    depth = 0.85 + 0.15 * (1.0 - np.abs(u - 0.5) * 0.4)
    ripple = 0.94 + 0.06 * np.sin(ys * 0.07 + t * 2.2)
    band = base[None, :] * (depth * ripple)[:, None]
    frame[:, :] = np.clip(band[:, None, :], 0, 255)

    for cu in currents:
        age = t - cu["birth"]
        if age < -0.05:
            continue
        span = W + cu["len"]
        x_head = (age * 160 * abs(cu["v_x"]) + cu["seed"] * span) % span
        if sx < 0:
            x_head = W - (x_head % (W + 1))
        segs = 40
        for i in range(segs):
            frac = i / segs
            x = x_head - sx * cu["len"] * frac
            y = cu["y0"] + math.sin(x * 0.02 + cu["phase"] + t * 4.2) * cu["amp_wave"]
            if y < y_top + 4 or y > y_bot - 4:
                continue
            fade = (1.0 - frac) * (0.55 + 0.45 * cu["amp"])
            # яркий тонкий штрих — читается как ток, не как blob
            tw = max(0.7, cu["thick"] * 0.7)
            col = RIVER_FOAM if frac < 0.12 else RIVER_SPEC
            add_blob(frame, x, y, tw * 3.5, tw * 0.22, col, 0.55 * fade)
            if i % 2 == 0:
                stroke_line(
                    frame, x, y, x - sx * 22, y + math.sin(cu["phase"] + i * 0.25) * 0.8,
                    RIVER_SHEET, 0.7, 0.4 * fade,
                )

    for f in foam:
        age = t - f["t0"]
        if age < 0 or age > 1.8:
            continue
        x = f["x0"] + f["v"] * age
        y = f["y"] + math.sin(age * 7 + f["phase"]) * 2.5
        if y < y_top + 4 or y > y_bot - 4:
            continue
        fade = math.exp(-age * 1.0) * f["amp"]
        stroke_line(frame, x - 14, y, x + 14, y, RIVER_FOAM, 0.85, 0.7 * fade)
        stroke_line(frame, x - 8, y + 1.2, x + 10, y - 0.6, RIVER_SPEC, 0.55, 0.35 * fade)


# ── ДОЖДЬ: тонкие падающие штрихи + корона удара ─────────────────────

RAIN_STREAK = (210, 185, 130)   # BGR холодная капля
RAIN_CORE = (235, 220, 200)
RAIN_RIPPLE = (180, 160, 110)
RAIN_CROWN = (230, 225, 215)


def render_rain(frame, t, drops, W, H):
    """Материя: диагональный штрих полёта + корона/кольца удара.
    Запрет: струи реки, ленты водопада, волокна ветра.
    """
    frame[:, :] = (14, 12, 11)
    ground = int(H * 0.82)
    frame[ground:, :] = (22, 18, 14)

    for d in drops:
        birth = d["birth"]
        amp = d["amp"]
        flight = max(0.18, 0.28 + 0.2 * (1.0 - amp))
        period = float(d.get("period") or 0.55)
        age = (t - birth) % period
        if age > flight + 0.85:
            continue
        x_imp = d["x"]
        y_imp = d["y_imp"]
        if age <= flight:
            u = age / flight
            x = x_imp + d["lean"] * (1.0 - u)
            y = -30 + (y_imp + 30) * (u ** 0.92)
            x0 = x - d["lean"] * 0.12
            y0 = y - (22 + amp * 16)
            stroke_line(frame, x0, y0, x, y, RAIN_STREAK, 0.9 + amp * 0.8, 0.55 + 0.35 * amp)
            stroke_line(frame, x0 * 0.3 + x * 0.7, y0 * 0.3 + y * 0.7, x, y, RAIN_CORE, 0.55, 0.7 * amp)
            add_blob(frame, x, y, 1.4, 2.0, RAIN_CORE, 0.7 * amp, hard=True)
        else:
            imp_age = age - flight
            fade = math.exp(-imp_age * 3.5) * amp
            if imp_age < 0.32:
                for ang in (-60, -30, 0, 30, 60):
                    rad = math.radians(ang)
                    L = (10 + amp * 16) * (1 - imp_age / 0.32) * fade
                    stroke_line(
                        frame, x_imp, y_imp,
                        x_imp + L * math.sin(rad), y_imp - L * math.cos(rad),
                        RAIN_CROWN, 0.85, 0.65 * fade,
                    )
            for k, scale in enumerate((1.0, 2.4, 4.0)):
                rr = (3 + amp * 7) * scale * (0.35 + imp_age * 1.5)
                aa = fade * (0.55 - k * 0.14)
                if aa < 0.05:
                    continue
                col = tuple(int(c * min(1.0, aa * 1.2)) for c in RAIN_RIPPLE)
                cv2.ellipse(
                    frame,
                    (int(x_imp), int(y_imp)),
                    (max(2, int(rr)), max(1, int(rr * 0.32))),
                    0, 0, 360, col, 1, lineType=cv2.LINE_AA,
                )


def finalize(frame, glow_sigma=2.0, mix=0.18):
    out = np.clip(frame, 0, 255).astype(np.uint8)
    if glow_sigma > 0:
        glow = cv2.GaussianBlur(out, (0, 0), glow_sigma)
        out = cv2.addWeighted(out, 1.0 - mix, glow, mix, 0)
    return out


def matter_id(kind: str) -> dict[str, Any]:
    """Поля для образ_причины.рендер."""
    table = {
        "ветер": {
            "материя": "broken_fiber_fleck",
            "тип": "wind_fiber_v2",
            "запрещено_материя": ["thick_blob_ribbon", "waterfall_ribbon", "river_sheet"],
        },
        "водопад": {
            "материя": "vertical_ribbon_hard_spray",
            "тип": "waterfall_ribbon_v2",
            "запрещено_материя": ["wind_fiber", "river_sheet", "identical_blob_chain"],
        },
        "река": {
            "материя": "current_filaments_foam",
            "тип": "river_current_v3",
            "запрещено_материя": ["ellipse_blob_lens", "wind_fiber", "waterfall_ribbon", "W_scale_gaussian"],
        },
        "дождь": {
            "материя": "drop_streak_impact_crown",
            "тип": "rain_drop_v1",
            "запрещено_материя": ["river_current", "wind_fiber", "waterfall_ribbon", "landscape_crop"],
        },
        "огонь": {
            "материя": "flame_tongue_crackle_spark",
            "тип": "flame_multilayer_v3",
            "запрещено_материя": ["wind_fiber", "river_sheet", "drop_streak", "hud_bowl"],
        },
        "гром": {
            "материя": "flash_then_shock_field",
            "тип": "flash_then_shock_draw",
            "запрещено_материя": ["rain_drop", "fire_tongue", "delay_fake_sync"],
        },
    }
    return table[kind]

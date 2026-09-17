# -*- coding: utf-8 -*-
"""Кусок №2: ВСЕ атомы клетки = звук_ядро + образ_причины (контракт).

Не выборка 12 — полная клетка etalon_dozhd.
Рендер: закон достаточен для детерминированной отрисовки (png на атом не плодим).
Геометрия: ближайшее событие/точка трека по времени, иначе approx.

Запуск: python3 scripts/собрать_полные_атомы_дождь_звук_образ.py
"""
from __future__ import annotations

import json
import math
import os
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")
TRACK = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.json")
EVENTS = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.json")
PARTICLE_PNG = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "частица_атом_масштаб.png")
OUT_JSON = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "атомы_звук_образ.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "атомы_полные_дождь_звук_образ.md")
OUT_META = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "meta.json")

R_REF = 0.001
RHO = 1000.0
VIDEO = "данные/стихии_живые/dozhd/video_live_01.mp4"
WAV = "данные/клеточки/эталоны/dozhd_real.wav"


def terminal_v(r_m: float) -> float:
    d_mm = 2 * r_m * 1000.0
    return float(min(9.5, max(0.8, 0.2 + 4.0 * (d_mm ** 0.7))))


def звук_ядро(a: dict, idx: int) -> dict:
    """Все звуковые параметры атома клетки (как есть) + явные ядровые поля."""
    # копируем все поля атома как звуковой паспорт клетки
    core = {k: v for k, v in a.items() if k not in ("образ_причины", "обратимость")}
    return {
        "atom_ref": f"etalon_dozhd#{idx}",
        "клетка": "etalon_dozhd",
        "звук_путь": WAV,
        "params_104": a.get("params_104"),  # честно: в клетке часто null
        "ядро": {
            "birth": a.get("birth"),
            "freq": a.get("freq"),
            "amp": a.get("amp"),
            "phase": a.get("phase"),
            "harmonicity": a.get("harmonicity"),
            "size": a.get("size"),
            "lifetime": a.get("lifetime"),
            "harmonic_index": a.get("harmonic_index"),
            "pos_x": a.get("pos_x"),
            "pos_y": a.get("pos_y"),
            "pos_z": a.get("pos_z"),
            "color_r": a.get("color_r"),
            "color_g": a.get("color_g"),
            "color_b": a.get("color_b"),
        },
        "поля_клетки": core,
    }


def образ_причины(a: dict, size_med: float, geo_points: list, events: list) -> dict:
    size_a = float(a.get("size") or size_med)
    r_m = float(max(2e-4, min(0.0025, (size_a / size_med) * R_REF)))
    v_ms = terminal_v(r_m)
    m = (4.0 / 3.0) * math.pi * (r_m ** 3) * RHO
    e_kin = 0.5 * m * v_ms ** 2
    birth = float(a.get("birth") or 0.0)

    # геометрия: ближайшая точка трека или событие (ось клипа ≈ birth при assumed_t0 в окне клетки)
    geo = {
        "video": VIDEO,
        "роль": "намёк места/момента, не весь пейзаж",
        "approx": True,
        "alignment_note": "birth на оси dozhd_real; video_live_01 — геометрия-намёк при assumed_t0",
    }
    best = None
    best_dt = 1e9
    for p in geo_points:
        dt = abs(float(p.get("t_sec") or 0) - birth)
        if dt < best_dt:
            best_dt = dt
            best = ("track", p)
    for e in events:
        dt = abs(float(e.get("t_sec") or 0) - birth)
        # события на длинном клипе: modulo/близость только если dt мало на оси клипа —
        # для birth∈[0,2.4] ищем события в том же окне
        if float(e.get("t_sec") or 0) < 2.5 and dt < best_dt:
            best_dt = dt
            best = ("event", e)
    if best and best_dt <= 0.08:
        kind, obj = best
        geo["approx"] = False
        geo["match_dt"] = round(best_dt, 4)
        geo["match_kind"] = kind
        if kind == "track":
            geo.update({
                "frame_i": obj.get("frame_i"),
                "t_sec": obj.get("t_sec"),
                "xy": [obj.get("x"), obj.get("y")],
                "track_ref": "отчёты/трек_капли_дождь.json",
            })
        else:
            geo.update({
                "frame_i": obj.get("frame_i"),
                "t_sec": obj.get("t_sec"),
                "bbox": obj.get("bbox"),
                "score": obj.get("score"),
                "event_id": obj.get("id"),
            })
    else:
        geo["t_sec"] = birth
        geo["xy"] = None
        geo["match_dt"] = None

    return {
        "событие": "drop_impact",
        "фаза": "both",
        "геометрия": geo,
        "закон": {
            "модель": "particle_scale_impulse",
            "r_m": r_m,
            "r_mm": round(r_m * 1000, 3),
            "v_ms": round(v_ms, 3),
            "m_kg": m,
            "e_kin_j": e_kin,
            "формула_масштаба": "r_m = (atom.size / size_med) * 1mm",
            "size_atom": size_a,
            "size_med": size_med,
            "surface": "water",
            "Minnaert": False,
            "note": "E: импульс ближе; пузырёк не обязателен",
        },
        "рендер": {
            "тип": "particle_draw",
            "детерминизм": "из закон.r_m + size; файл не обязателен",
            "path_шаблон": os.path.relpath(PARTICLE_PNG, КОРЕНЬ) if os.path.isfile(PARTICLE_PNG) else None,
            "запрещено": ["landscape_crop", "willow_canopy", "full_frame_as_atom", "hud_circles_on_video"],
        },
        "не_есть": "кроп листвы/облака/лужи целиком",
    }


def main() -> int:
    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms_in = cell["atoms"]
    sizes = [float(a.get("size") or 0.11) for a in atoms_in]
    size_med = float(np.median(sizes))

    geo_points = []
    if os.path.isfile(TRACK):
        geo_points = json.load(open(TRACK, encoding="utf-8")).get("points") or []
    events = []
    if os.path.isfile(EVENTS):
        events = json.load(open(EVENTS, encoding="utf-8")).get("events") or []

    full = []
    n_geo_exact = 0
    for i, a in enumerate(atoms_in):
        img = образ_причины(a, size_med, geo_points, events)
        if not img["геометрия"].get("approx", True):
            n_geo_exact += 1
        full.append({
            "id": f"dozhd_full_{i:05d}",
            "стихия": "дождь",
            "birth": a.get("birth"),
            "alignment": "assumed_t0",
            "звук_ядро": звук_ядро(a, i),
            "образ_причины": img,
            "обратимость": {
                "звук_в_петле": True,
                "образ_записан_в_атом": True,
                "образ_в_синтезе_визуала": False,
                "геометрия_approx": bool(img["геометрия"].get("approx", True)),
                "params_104_на_атоме": a.get("params_104") is not None,
                "note": "полный атом: поля звука клетки + образ_причины; визуал-сборка — кусок №3",
            },
        })

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    package = {
        "дата": date.today().isoformat(),
        "контракт": "отчёты/контракт_образ_в_атоме_дождь.md",
        "клетка": "etalon_dozhd",
        "n_atoms": len(full),
        "size_med": size_med,
        "n_геометрия_не_approx": n_geo_exact,
        "crosses_ref": "в клетке etalon_dozhd.json (не дублируем)",
        "atoms": full,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False)

    meta = {
        "дата": date.today().isoformat(),
        "n_atoms": len(full),
        "size_med": size_med,
        "n_геометрия_не_approx": n_geo_exact,
        "доля_геометрии": round(n_geo_exact / max(len(full), 1), 4),
        "file": os.path.relpath(OUT_JSON, КОРЕНЬ),
        "sample_ids": [full[0]["id"], full[len(full) // 2]["id"], full[-1]["id"]],
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    lines = [
        "# Полные атомы дождь: звук + образ_причины",
        "",
        f"> **n = {len(full)}** (вся клетка, не 12) · геометрия точная у **{n_geo_exact}** · size_med={size_med:.4f}",
        "",
        f"- пакет: `{os.path.relpath(OUT_JSON, КОРЕНЬ)}`",
        f"- meta: `{os.path.relpath(OUT_META, КОРЕНЬ)}`",
        f"- контракт: `отчёты/контракт_образ_в_атоме_дождь.md`",
        "",
        "Каждый атом: `звук_ядро` (все поля клетки) + `образ_причины` (закон size→r + рендер particle_draw).",
        "`образ_в_синтезе_визуала: false` — кусок №3.",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({
        "ok": True,
        "n_atoms": len(full),
        "n_geo_exact": n_geo_exact,
        "out": OUT_JSON,
        "bytes": os.path.getsize(OUT_JSON),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

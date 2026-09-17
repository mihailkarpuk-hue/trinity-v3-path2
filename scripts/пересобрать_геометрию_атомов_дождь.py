# -*- coding: utf-8 -*-
"""Пересборка геометрии в пакете атомов после плотного детектора.

Сохраняет alignment axes_decoupled и t0_geometry_offset.
Для event-match пишет xy из cx,cy.
При нескольких кандидатах в thr — выбирает ближайший к (pos_x,pos_y) атома.

Запуск: python3 scripts/пересобрать_геометрию_атомов_дождь.py
"""
from __future__ import annotations

import json
import os
from datetime import date

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "атомы_звук_образ.json")
META = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "meta.json")
ALIGN = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "alignment_t0.json")
TRACK = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.json")
EVENTS = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "геометрия_атомов_плотная.md")

VIDEO_REL = "данные/стихии_живые/dozhd/video_live_01.mp4"
MATCH_THR = 0.08
# кадр video_live_01 ~1440x1080 — для нормализации pos атома
VW, VH = 1440.0, 1080.0


def atom_xy_hint(a: dict):
    """pos_x/y клетки ≈ [-2, 2.5] → нормировка в кадр видео."""
    core = (a.get("звук_ядро") or {}).get("ядро") or {}
    px, py = core.get("pos_x"), core.get("pos_y")
    if px is None or py is None:
        return None
    px, py = float(px), float(py)
    # мягкий clamp в [0,1] через сдвиг/масштаб типичного диапазона
    nx = (px + 2.0) / 4.5
    ny = (py + 2.0) / 4.5
    nx = max(0.05, min(0.95, nx))
    ny = max(0.40, min(0.95, ny))  # лужа — нижняя часть кадра
    return (nx * VW, ny * VH)


def cand_xy(kind: str, obj: dict) -> tuple[float, float] | None:
    if kind == "track":
        if obj.get("x") is None:
            return None
        return float(obj["x"]), float(obj["y"])
    if obj.get("cx") is not None:
        return float(obj["cx"]), float(obj["cy"])
    if obj.get("xy"):
        return float(obj["xy"][0]), float(obj["xy"][1])
    return None


def best_match(t_причина: float, hint: tuple[float, float] | None, geo_points: list, events: list):
    cands = []
    for p in geo_points:
        dt = abs(float(p.get("t_sec") or 0) - t_причина)
        if dt <= MATCH_THR:
            cands.append(("track", p, dt))
    for e in events:
        dt = abs(float(e.get("t_sec") or 0) - t_причина)
        if dt <= MATCH_THR:
            cands.append(("event", e, dt))
    if not cands:
        return None, None
    if hint is None:
        cands.sort(key=lambda t: t[2])
        return cands[0][0], cands[0][1]

    def dist(c):
        kind, obj, dt = c
        xy = cand_xy(kind, obj)
        if xy is None:
            return (dt, 1e9)
        return (dt, (xy[0] - hint[0]) ** 2 + (xy[1] - hint[1]) ** 2)

    cands.sort(key=dist)
    return cands[0][0], cands[0][1]


def main() -> int:
    pkg = json.load(open(PKG, encoding="utf-8"))
    atoms = pkg["atoms"]
    t0_geo = float(pkg.get("alignment", {}).get("geometry_t0_offset") or 1.6)
    if os.path.isfile(ALIGN):
        t0_geo = float(json.load(open(ALIGN, encoding="utf-8")).get("geometry_t0_offset") or t0_geo)

    geo_points = []
    if os.path.isfile(TRACK):
        geo_points = json.load(open(TRACK, encoding="utf-8")).get("points") or []
    events = []
    if os.path.isfile(EVENTS):
        events = json.load(open(EVENTS, encoding="utf-8")).get("events") or []

    n_exact = 0
    n_with_xy = 0
    for a in atoms:
        birth = float(a.get("birth") or 0.0)
        t_причина = birth + t0_geo
        a["alignment"] = "axes_decoupled"
        a["t_sec_причина"] = round(t_причина, 4)
        a["t0_geometry_offset"] = t0_geo
        a["video_причина"] = VIDEO_REL

        hint = atom_xy_hint(a)
        kind, obj = best_match(t_причина, hint, geo_points, events)
        geo = (a.get("образ_причины") or {}).get("геометрия") or {}
        geo = dict(geo)
        geo.update({
            "video": VIDEO_REL,
            "роль": "намёк места/момента, не весь пейзаж",
            "t0_geometry_offset": t0_geo,
            "alignment_note": (
                f"birth на dozhd_real; t_sec_причина=birth+{t0_geo:.3f}; "
                "плотная геометрия multi-blob"
            ),
            "t_sec": round(t_причина, 4),
        })
        if kind and obj:
            geo["approx"] = False
            geo["match_kind"] = kind
            dt = abs(float(obj.get("t_sec") or 0) - t_причина)
            geo["match_dt"] = round(dt, 4)
            geo["t_sec_match"] = obj.get("t_sec")
            xy = cand_xy(kind, obj)
            if xy:
                geo["xy"] = [int(xy[0]), int(xy[1])]
                n_with_xy += 1
            if kind == "track":
                geo["frame_i"] = obj.get("frame_i")
                geo["track_ref"] = "отчёты/трек_капли_дождь.json"
                geo.pop("bbox", None)
                geo.pop("event_id", None)
            else:
                geo["frame_i"] = obj.get("frame_i")
                geo["bbox"] = obj.get("bbox")
                geo["score"] = obj.get("score")
                geo["event_id"] = obj.get("id")
                geo["cx"] = obj.get("cx")
                geo["cy"] = obj.get("cy")
            n_exact += 1
        else:
            geo["approx"] = True
            geo["xy"] = None
            geo["match_dt"] = None
            geo["match_kind"] = None

        img = dict(a.get("образ_причины") or {})
        img["геометрия"] = geo
        a["образ_причины"] = img
        obr = a.get("обратимость") or {}
        obr["геометрия_approx"] = bool(geo.get("approx", True))
        a["обратимость"] = obr

    pkg["дата"] = date.today().isoformat()
    pkg["n_геометрия_не_approx"] = n_exact
    pkg["n_геометрия_с_xy"] = n_with_xy
    al = pkg.get("alignment") or {}
    al.update({
        "режим": "axes_decoupled",
        "geometry_t0_offset": t0_geo,
        "геометрия_плотная": True,
        "n_events_источник": len(events),
        "n_track_points": len(geo_points),
        "n_геометрия_не_approx": n_exact,
        "n_геометрия_с_xy": n_with_xy,
        "match_thr_s": MATCH_THR,
    })
    pkg["alignment"] = al
    with open(PKG, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False)

    meta = {}
    if os.path.isfile(META):
        meta = json.load(open(META, encoding="utf-8"))
    meta.update({
        "дата": date.today().isoformat(),
        "n_геометрия_не_approx": n_exact,
        "n_геометрия_с_xy": n_with_xy,
        "доля_геометрии": round(n_exact / max(len(atoms), 1), 4),
        "доля_xy": round(n_with_xy / max(len(atoms), 1), 4),
        "n_events": len(events),
        "n_track_points": len(geo_points),
    })
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    report = {
        "дата": date.today().isoformat(),
        "n_atoms": len(atoms),
        "n_exact": n_exact,
        "n_with_xy": n_with_xy,
        "доля_exact": round(n_exact / max(len(atoms), 1), 4),
        "n_events": len(events),
        "n_track_points": len(geo_points),
        "t0_geo": t0_geo,
        "thr": MATCH_THR,
    }
    lines = [
        "# Геометрия атомов после плотного детектора",
        "",
        f"> {report['дата']}",
        "",
        f"- точных match: **{n_exact}** / {len(atoms)} ({report['доля_exact']})",
        f"- с xy: **{n_with_xy}**",
        f"- событий: {len(events)} · точек трека: {len(geo_points)}",
        f"- t0_geo={t0_geo} · thr={MATCH_THR}s",
        "",
        f"Пакет: `{os.path.relpath(PKG, КОРЕНЬ)}`",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

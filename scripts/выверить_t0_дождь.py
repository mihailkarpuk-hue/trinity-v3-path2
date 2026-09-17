# -*- coding: utf-8 -*-
"""Выверка t0: dozhd_real ↔ video_live_01.

Факт (измерение): wav ≠ AAC клипа (NCC≈0.02) — оси звука и видео развязаны.
Политика: alignment = axes_decoupled + geometry_t0_offset (макс. совпадений
track/events с t_причина = birth + offset).

Запуск: python3 scripts/выверить_t0_дождь.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "атомы_звук_образ.json")
META = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "meta.json")
ALIGN_JSON = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "alignment_t0.json")
ALIGN_MD = os.path.join(КОРЕНЬ, "отчёты", "alignment_t0_дождь.md")
TRACK = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.json")
EVENTS = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.json")
WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "dozhd_real.wav")
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
VIDEO_REL = "данные/стихии_живые/dozhd/video_live_01.mp4"
PROBE_WAV = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "_live01_aac_t0.wav")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")

R_REF = 0.001
RHO = 1000.0
MATCH_THR = 0.08


def _ff() -> str:
    return FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"


def load_wav(path: str):
    with wave.open(path) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768.0
    return x, sr


def measure_ncc(real: np.ndarray, live: np.ndarray, sr: int) -> dict:
    win = len(real)
    step = max(1, int(0.02 * sr))
    realz = real - real.mean()
    rn = float(np.linalg.norm(realz) + 1e-12)
    best = (-1.0, 0.0)
    for off in range(0, max(1, len(live) - win), step):
        seg = live[off : off + win]
        segz = seg - seg.mean()
        ncc = float(np.dot(realz, segz) / (rn * (np.linalg.norm(segz) + 1e-12)))
        if ncc > best[0]:
            best = (ncc, off / sr)
    # refine ±40 ms
    c = int(best[1] * sr)
    best_f = best
    for off in range(c - int(0.04 * sr), c + int(0.04 * sr)):
        if off < 0 or off + win > len(live):
            continue
        seg = live[off : off + win]
        segz = seg - seg.mean()
        ncc = float(np.dot(realz, segz) / (rn * (np.linalg.norm(segz) + 1e-12)))
        if ncc > best_f[0]:
            best_f = (ncc, off / sr)
    return {
        "ncc_max": round(best_f[0], 5),
        "ncc_lag_s": round(best_f[1], 4),
        "verdict": "axes_decoupled" if best_f[0] < 0.15 else "possible_same_take",
        "порог_совпадения": 0.15,
        "note": "NCC≪порога → dozhd_real не вырезка AAC video_live_01",
    }


def terminal_v(r_m: float) -> float:
    d_mm = 2 * r_m * 1000.0
    return float(min(9.5, max(0.8, 0.2 + 4.0 * (d_mm ** 0.7))))


def match_geo(t_причина: float, geo_points: list, events: list):
    best = None
    best_dt = 1e9
    for p in geo_points:
        dt = abs(float(p.get("t_sec") or 0) - t_причина)
        if dt < best_dt:
            best_dt = dt
            best = ("track", p)
    for e in events:
        dt = abs(float(e.get("t_sec") or 0) - t_причина)
        if dt < best_dt:
            best_dt = dt
            best = ("event", e)
    if best and best_dt <= MATCH_THR:
        return best, best_dt
    return None, best_dt


def score_offset(births: list, offset: float, geo_points: list, events: list) -> int:
    n = 0
    for b in births:
        hit, _ = match_geo(b + offset, geo_points, events)
        if hit:
            n += 1
    return n


def rebuild_образ(a_atom: dict, size_med: float, t0_geo: float, geo_points: list, events: list) -> dict:
    """Обновить геометрию; закон/рендер сохранить если есть."""
    img = dict(a_atom.get("образ_причины") or {})
    birth = float(a_atom.get("birth") or 0.0)
    t_причина = birth + t0_geo
    size_a = float(
        (img.get("закон") or {}).get("size_atom")
        or (a_atom.get("звук_ядро") or {}).get("ядро", {}).get("size")
        or size_med
    )
    r_m = float(max(2e-4, min(0.0025, (size_a / size_med) * R_REF)))
    v_ms = terminal_v(r_m)
    m = (4.0 / 3.0) * math.pi * (r_m ** 3) * RHO
    e_kin = 0.5 * m * v_ms ** 2

    geo = {
        "video": VIDEO_REL,
        "роль": "намёк места/момента, не весь пейзаж",
        "approx": True,
        "alignment_note": (
            f"birth на оси dozhd_real; t_sec_причина = birth + {t0_geo:.3f} "
            "(геометрия video_live_01; звук≠AAC)"
        ),
        "t0_geometry_offset": t0_geo,
    }
    hit, best_dt = match_geo(t_причина, geo_points, events)
    if hit:
        kind, obj = hit
        geo["approx"] = False
        geo["match_dt"] = round(best_dt, 4)
        geo["match_kind"] = kind
        geo["t_sec"] = round(t_причина, 4)
        if kind == "track":
            geo.update({
                "frame_i": obj.get("frame_i"),
                "xy": [obj.get("x"), obj.get("y")],
                "track_ref": "отчёты/трек_капли_дождь.json",
                "t_sec_match": obj.get("t_sec"),
            })
        else:
            geo.update({
                "frame_i": obj.get("frame_i"),
                "bbox": obj.get("bbox"),
                "score": obj.get("score"),
                "event_id": obj.get("id"),
                "t_sec_match": obj.get("t_sec"),
            })
    else:
        geo["t_sec"] = round(t_причина, 4)
        geo["xy"] = None
        geo["match_dt"] = None

    law = img.get("закон") or {
        "модель": "particle_scale_impulse",
        "формула_масштаба": "r_m = (atom.size / size_med) * 1mm",
        "surface": "water",
        "Minnaert": False,
        "note": "E: импульс ближе; пузырёк не обязателен",
    }
    law = dict(law)
    law.update({
        "r_m": r_m,
        "r_mm": round(r_m * 1000, 3),
        "v_ms": round(v_ms, 3),
        "m_kg": m,
        "e_kin_j": e_kin,
        "size_atom": size_a,
        "size_med": size_med,
    })
    render = img.get("рендер") or {
        "тип": "particle_draw",
        "детерминизм": "из закон.r_m + size; файл не обязателен",
        "запрещено": ["landscape_crop", "willow_canopy", "full_frame_as_atom", "hud_circles_on_video"],
    }
    return {
        "событие": img.get("событие") or "drop_impact",
        "фаза": img.get("фаза") or "both",
        "геометрия": geo,
        "закон": law,
        "рендер": render,
        "не_есть": img.get("не_есть") or "кроп листвы/облака/лужи целиком",
    }


def main() -> int:
    if not os.path.isfile(PKG):
        raise SystemExit(f"нет пакета: {PKG}")

    # 1) AAC probe + NCC
    os.makedirs(os.path.dirname(PROBE_WAV), exist_ok=True)
    subprocess.run(
        [_ff(), "-y", "-i", VIDEO, "-t", "30", "-ac", "1", "-ar", "22050", PROBE_WAV],
        capture_output=True,
        check=False,
    )
    real, sr = load_wav(WAV)
    live, _ = load_wav(PROBE_WAV)
    audio_meas = measure_ncc(real, live, sr)

    geo_points = []
    if os.path.isfile(TRACK):
        geo_points = json.load(open(TRACK, encoding="utf-8")).get("points") or []
    events = []
    if os.path.isfile(EVENTS):
        events = json.load(open(EVENTS, encoding="utf-8")).get("events") or []

    pkg = json.load(open(PKG, encoding="utf-8"))
    atoms = pkg["atoms"]
    births = [float(a.get("birth") or 0.0) for a in atoms]
    size_med = float(pkg.get("size_med") or 0.1133)

    # 2) sweep geometry offset
    scores = []
    for off in np.arange(0.0, 22.0, 0.05):
        scores.append((score_offset(births, float(off), geo_points, events), round(float(off), 3)))
    scores.sort(reverse=True)
    n0 = score_offset(births, 0.0, geo_points, events)
    t0_geo = scores[0][1]
    n_best = scores[0][0]

    # 3) rewrite atoms
    n_exact = 0
    for a in atoms:
        birth = float(a.get("birth") or 0.0)
        t_причина = birth + t0_geo
        a["alignment"] = "axes_decoupled"
        a["t_sec_причина"] = round(t_причина, 4)
        a["video_причина"] = VIDEO_REL
        a["t0_geometry_offset"] = t0_geo
        img = rebuild_образ(a, size_med, t0_geo, geo_points, events)
        a["образ_причины"] = img
        approx = bool(img["геометрия"].get("approx", True))
        if not approx:
            n_exact += 1
        obr = a.get("обратимость") or {}
        obr["геометрия_approx"] = approx
        obr["alignment"] = "axes_decoupled"
        a["обратимость"] = obr

    pkg["дата"] = date.today().isoformat()
    pkg["n_геометрия_не_approx"] = n_exact
    pkg["alignment"] = {
        "режим": "axes_decoupled",
        "audio": audio_meas,
        "geometry_t0_offset": t0_geo,
        "n_геометрия_при_offset0": n0,
        "n_геометрия_при_offset": n_exact,
        "match_thr_s": MATCH_THR,
        "top_offsets": [{"n": n, "offset": o} for n, o in scores[:8]],
        "формула": "t_sec_причина = birth + geometry_t0_offset",
        "звук_ось": "dozhd_real / клетка",
        "видео_ось": VIDEO_REL,
    }
    with open(PKG, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False)

    align_report = {
        "дата": date.today().isoformat(),
        **pkg["alignment"],
        "n_atoms": len(atoms),
        "доля_геометрии": round(n_exact / max(len(atoms), 1), 4),
        "package": os.path.relpath(PKG, КОРЕНЬ),
    }
    with open(ALIGN_JSON, "w", encoding="utf-8") as f:
        json.dump(align_report, f, ensure_ascii=False, indent=2)

    meta = {}
    if os.path.isfile(META):
        meta = json.load(open(META, encoding="utf-8"))
    meta.update({
        "дата": date.today().isoformat(),
        "n_геометрия_не_approx": n_exact,
        "доля_геометрии": round(n_exact / max(len(atoms), 1), 4),
        "alignment": "axes_decoupled",
        "geometry_t0_offset": t0_geo,
        "audio_ncc_max": audio_meas["ncc_max"],
    })
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    lines = [
        "# Alignment t0 — дождь",
        "",
        f"> {date.today().isoformat()} · `assumed_t0` снят",
        "",
        "## Вердикт",
        "",
        f"- **Звук ↔ AAC:** NCC_max = **{audio_meas['ncc_max']}** @ lag {audio_meas['ncc_lag_s']}s "
        f"→ **{audio_meas['verdict']}** (порог {audio_meas['порог_совпадения']}).",
        "- `dozhd_real` **не** вырезка `video_live_01`. Синхрон «один клип» невозможен.",
        f"- **Геометрия:** `t_sec_причина = birth + {t0_geo}` "
        f"(было offset 0 → {n0} точных; стало → **{n_exact}** / {len(atoms)}).",
        "",
        "## Режим",
        "",
        "| Поле | Значение |",
        "|------|----------|",
        "| `alignment` | `axes_decoupled` |",
        f"| `geometry_t0_offset` | {t0_geo} |",
        f"| `match_thr` | {MATCH_THR}s |",
        "| звуковая ось | `dozhd_real` / клетка |",
        f"| видео-ось | `{VIDEO_REL}` |",
        "",
        f"- отчёт json: `{os.path.relpath(ALIGN_JSON, КОРЕНЬ)}`",
        f"- пакет обновлён: `{os.path.relpath(PKG, КОРЕНЬ)}`",
        "",
    ]
    with open(ALIGN_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(align_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

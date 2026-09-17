# -*- coding: utf-8 -*-
"""Полные атомы огня: звук_ядро + образ_причины combustion_crackle.

Событие = микро-треск горения + подъём пламени (buoyancy), не костёр-пейзаж.
ogon_real ↔ video NCC≈0.04 → axes_decoupled; t_sec_причина = birth (звук),
геометрия видео — только намёк яркости пламени.

Запуск: python3 scripts/собрать_полные_атомы_огонь_звук_образ.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро")]

from кресты import построить_кресты, сводка_решетки  # noqa: E402

КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_ogon.json")
WAV = "данные/клеточки/эталоны/ogon_real.wav"
VID = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "ogon", "clean_video.mp4")
VID_REL = "данные/стихии_живые/ogon/clean_video.mp4"
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь")
OUT_JSON = os.path.join(OUT_DIR, "атомы_звук_образ.json")
OUT_CROSSES = os.path.join(OUT_DIR, "кресты.json")
OUT_META = os.path.join(OUT_DIR, "meta.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "атомы_полные_огонь_звук_образ.md")
CONTRACT = os.path.join(КОРЕНЬ, "отчёты", "контракт_образ_в_атоме_огонь.md")


def flame_hint(video: str) -> dict:
    """Намёк: centroid яркого пламени (не кадр как атом)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {"xy": None, "approx": True, "note": "видео не открылось"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    acc = None
    n = 0
    fi = -1
    best = (-1.0, 0, None)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        if fi % 3 != 0:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        thr = np.percentile(g, 97)
        mask = g >= thr
        if acc is None:
            acc = mask.astype(np.float64)
        else:
            acc += mask
        n += 1
        m = float(g.mean())
        if m > best[0]:
            ys, xs = np.where(mask)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean())
            else:
                h, w = g.shape
                cx, cy = w // 2, h // 2
            best = (m, fi, (cx, cy))
    cap.release()
    h = w = None
    base_xy = None
    if acc is not None and n > 0:
        heat = acc / n
        h, w = heat.shape
        # основание пламени = нижняя треть яркого облака
        ys, xs = np.where(heat > np.percentile(heat, 90))
        if len(xs):
            cx = int(xs.mean())
            # низ яркого пятна
            cy = int(np.percentile(ys, 75))
            base_xy = [cx, cy]
    return {
        "xy_peak": list(best[2]) if best[2] else None,
        "xy_base": base_xy,
        "frame_peak": int(best[1]),
        "t_peak": round(best[1] / fps, 4) if fps else None,
        "fps": fps,
        "wh": [int(w), int(h)] if w else None,
        "approx": True,
        "note": "намёк яркости пламени; wav≠video AAC → не синхрон",
    }


def звук_ядро(a: dict, idx: int) -> dict:
    core = {k: v for k, v in a.items()}
    return {
        "atom_ref": f"etalon_ogon#{idx}",
        "клетка": "etalon_ogon",
        "звук_путь": WAV,
        "params_104": a.get("params_104"),
        "долг_params_104": a.get("params_104") is None,
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


def образ_причины(a: dict, hint: dict, size_med: float) -> dict:
    birth = float(a.get("birth") or 0.0)
    size_a = float(a.get("size") or size_med)
    amp = float(a.get("amp") or 0.1)
    energy_rel = size_a / max(size_med, 1e-9)
    # подъём: чем крупнее/громче — тем быстрее вверх (buoyancy proxy)
    v_up = float(np.clip(0.15 + 0.55 * energy_rel + 0.2 * amp, 0.1, 1.2))
    return {
        "событие": "combustion_crackle",
        "фаза": "both",
        "треск": {
            "звук": WAV,
            "t_crackle": birth,
            "broadband": True,
            "note": "импульс треска на оси звука",
        },
        "пламя": {
            "video_намёк": VID_REL,
            "xy_base": hint.get("xy_base"),
            "xy_peak": hint.get("xy_peak"),
            "approx": True,
            "роль": "направление подъёма, не кадр костра",
        },
        "закон": {
            "модель": "buoyancy_rise_crackle",
            "формула": "v_up ≈ k*(size/size_med) + c*amp; crackle@birth",
            "v_up": round(v_up, 4),
            "energy_rel": round(energy_rel, 4),
            "size_atom": size_a,
            "size_med": size_med,
            "delay_s": None,
            "approx": True,
            "note": "закон подъёма из звуковых size/amp; sync с видео нет",
        },
        "геометрия": {
            "video": VID_REL,
            "t_sec": birth,
            "xy": hint.get("xy_base") or hint.get("xy_peak"),
            "approx": True,
            "note": "координата намёка, не crop пейзажа",
        },
        "рендер": {
            "тип": "flame_tongue_crackle_draw",
            "запрещено": [
                "landscape_campfire",
                "bowl_firepit",
                "full_frame_as_atom",
                "hud_circles",
            ],
        },
        "не_есть": "костёр как пейзаж, чаша, полный кадр пламени, HUD-кружки",
    }


def main() -> int:
    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms_in = cell.get("atoms") or cell.get("атомы") or []
    sizes = [float(a.get("size") or 0.11) for a in atoms_in]
    size_med = float(np.median(sizes))

    hint = flame_hint(VID) if os.path.isfile(VID) else {"approx": True}

    crosses_raw = cell.get("crosses") or []
    if not crosses_raw:
        crosses_raw = построить_кресты(atoms_in)
    решетка = сводка_решетки(atoms_in, crosses_raw)

    full = []
    adj = defaultdict(list)
    crosses = []
    for i, c in enumerate(crosses_raw):
        ia, ib = int(c["atom_a"]), int(c["atom_b"])
        id_a, id_b = f"ogon_full_{ia:05d}", f"ogon_full_{ib:05d}"
        edge = {
            "i": i,
            "atom_a": ia,
            "atom_b": ib,
            "id_a": id_a,
            "id_b": id_b,
            "axis": c.get("axis"),
            "direction": c.get("direction"),
            "type": c.get("type"),
            "resonance": c.get("resonance"),
            "cross_type": c.get("cross_type"),
            "energy_flow": c.get("energy_flow"),
            "master_cross_face": c.get("master_cross_face"),
        }
        crosses.append(edge)
        compact = {
            "cross_i": i,
            "axis": edge["axis"],
            "resonance": edge["resonance"],
            "cross_type": edge["cross_type"],
            "face": edge["master_cross_face"],
            "direction": edge["direction"],
            "energy_flow": edge["energy_flow"],
        }
        adj[ia].append({**compact, "к": id_b, "к_idx": ib})
        adj[ib].append({**compact, "к": id_a, "к_idx": ia})

    for i, a in enumerate(atoms_in):
        img = образ_причины(a, hint, size_med)
        links = adj.get(i) or []
        birth = float(a.get("birth") or 0.0)
        full.append({
            "id": f"ogon_full_{i:05d}",
            "стихия": "огонь",
            "birth": a.get("birth"),
            "alignment": "axes_decoupled",
            "t_sec_причина": birth,  # причина на звуковой оси; видео не синхрон
            "звук_ядро": звук_ядро(a, i),
            "образ_причины": img,
            "кресты": {"n": len(links), "связи": links},
            "обратимость": {
                "звук_в_петле": True,
                "образ_записан_в_атом": True,
                "образ_в_синтезе_визуала": False,
                "геометрия_approx": True,
                "params_104_на_атоме": a.get("params_104") is not None,
                "кресты_на_атоме": True,
                "note": "огонь: звук из клетки; видео — намёк; NCC≠sync",
            },
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    package = {
        "дата": date.today().isoformat(),
        "контракт": os.path.relpath(CONTRACT, КОРЕНЬ),
        "клетка": "etalon_ogon",
        "n_atoms": len(full),
        "size_med": size_med,
        "событие": "combustion_crackle",
        "flame_hint": hint,
        "alignment": {
            "режим": "axes_decoupled",
            "звук_ось": WAV,
            "video_ось": VID_REL,
            "NCC_note": "~0.04 wav vs clean_video AAC → не синхрон",
        },
        "crosses": crosses,
        "число_связей": len(crosses),
        "решетка": решетка,
        "crosses_построены_для_пакета": not bool(cell.get("crosses")),
        "atoms": full,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False)

    with open(OUT_CROSSES, "w", encoding="utf-8") as f:
        json.dump({
            "дата": date.today().isoformat(),
            "n_crosses": len(crosses),
            "решетка": решетка,
            "пересобраны": not bool(cell.get("crosses")),
            "crosses": crosses,
        }, f, ensure_ascii=False)

    meta = {
        "дата": date.today().isoformat(),
        "n_atoms": len(full),
        "n_crosses": len(crosses),
        "size_med": size_med,
        "событие": "combustion_crackle",
        "alignment": "axes_decoupled",
        "file": os.path.relpath(OUT_JSON, КОРЕНЬ),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    contract = f"""# Контракт: образ в атоме огня (combustion_crackle)

> {date.today().isoformat()} · третий эталон после дождя и грома  
> Звук: `ogon_real` / `etalon_ogon`.

## Событие

`combustion_crackle` = микро-треск горения (звук @ birth) + подъём языка пламени (buoyancy).

Закон: `v_up ≈ k*(size/size_med) + c*amp`; треск на звуковой оси.

## Честность

`ogon_real.wav` и `clean_video` AAC — **не синхрон** (NCC≈0.04).  
Alignment: `axes_decoupled`. Геометрия видео — **намёк**, не crop пейзажа.

## Запреты

landscape campfire / bowl / full frame / HUD circles.

## Намёк пламени

xy_base={hint.get('xy_base')} xy_peak={hint.get('xy_peak')} t_peak={hint.get('t_peak')}
"""
    with open(CONTRACT, "w", encoding="utf-8") as f:
        f.write(contract)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(
            f"# Полные атомы огонь: звук + combustion_crackle\n\n"
            f"> n={len(full)} · crosses={len(crosses)} · axes_decoupled\n\n"
            f"- пакет: `{os.path.relpath(OUT_JSON, КОРЕНЬ)}`\n"
            f"- контракт: `{os.path.relpath(CONTRACT, КОРЕНЬ)}`\n"
        )

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

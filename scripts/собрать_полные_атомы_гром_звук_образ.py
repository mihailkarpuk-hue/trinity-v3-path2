# -*- coding: utf-8 -*-
"""Полные атомы грома: звук_ядро + образ_причины flash_shockwave.

Честно: molniya и grom_real — разные исходники → delay_s=null, закон approx.
Молния = момент вспышки (намёк), гром = звуковая ось birth.

Запуск: python3 scripts/собрать_полные_атомы_гром_звук_образ.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро")]

from кресты import построить_кресты, сводка_решетки  # noqa: E402

КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_grom.json")
WAV = "данные/клеточки/эталоны/grom_real.wav"
MOL_VID = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "molniya", "video_live_01.mp4")
MOL_VID_REL = "данные/стихии_живые/molniya/video_live_01.mp4"
GROM_VID_REL = "данные/стихии_живые/grom/video_live_01.mp4"
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_гром")
OUT_JSON = os.path.join(OUT_DIR, "атомы_звук_образ.json")
OUT_CROSSES = os.path.join(OUT_DIR, "кресты.json")
OUT_META = os.path.join(OUT_DIR, "meta.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "атомы_полные_гром_звук_образ.md")
CONTRACT = os.path.join(КОРЕНЬ, "отчёты", "контракт_образ_в_атоме_гром.md")


def detect_flash_t(video: str) -> dict:
    """Момент вспышки = пик яркости (намёк геометрии)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {"t_flash": None, "approx": True, "note": "видео не открылось"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    best = (-1.0, 0, None)  # mean, frame, (cx,cy) bright centroid
    fi = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        m = float(g.mean())
        if m > best[0]:
            # centroid of top 1% bright
            thr = np.percentile(g, 99)
            ys, xs = np.where(g >= thr)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean())
            else:
                h, w = g.shape
                cx, cy = w // 2, h // 2
            best = (m, fi, (cx, cy))
    cap.release()
    if best[1] < 0:
        return {"t_flash": None, "approx": True}
    return {
        "t_flash": round(best[1] / fps, 4),
        "frame_i": int(best[1]),
        "xy": list(best[2]),
        "brightness_mean": round(best[0], 2),
        "fps": fps,
        "approx": True,  # другая запись, не синхрон с grom_real
        "note": "вспышка с molniya; не синхронна с grom_real → delay не считается",
    }


def звук_ядро(a: dict, idx: int) -> dict:
    core = {k: v for k, v in a.items()}
    return {
        "atom_ref": f"etalon_grom#{idx}",
        "клетка": "etalon_grom",
        "звук_путь": WAV,
        "params_104": a.get("params_104"),
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


def образ_причины(a: dict, flash: dict, size_med: float) -> dict:
    birth = float(a.get("birth") or 0.0)
    size_a = float(a.get("size") or size_med)
    # масштаб «силы» раската ∝ size (не радиус капли)
    energy_rel = size_a / max(size_med, 1e-9)
    return {
        "событие": "flash_shockwave",
        "фаза": "both",
        "вспышка": {
            "источник": MOL_VID_REL,
            "t_flash": flash.get("t_flash"),
            "frame_i": flash.get("frame_i"),
            "xy": flash.get("xy"),
            "роль": "момент и намёк направления, не образ сам по себе",
            "approx": True,
            "note": flash.get("note"),
        },
        "раскат": {
            "звук": WAV,
            "t_thunder": birth,
            "video_намёк": GROM_VID_REL,
            "delay_s": None,  # нет синхронной пары molniya↔grom_real
            "delay_note": "разные исходники каталога → не выдумывать задержку",
        },
        "закон": {
            "модель": "shockwave_delay",
            "формула": "distance_km = delay_s * 0.343",
            "delay_s": None,
            "distance_km": None,
            "approx": True,
            "energy_rel": round(energy_rel, 4),
            "size_atom": size_a,
            "size_med": size_med,
            "note": "закон применим только при синхронной паре вспышка+гром; сейчас approx",
        },
        "рендер": {
            "тип": "flash_then_shock_draw",
            "запрещено": ["landscape_storm", "full_sky_as_atom", "decorative_flash_without_physics"],
        },
        "не_есть": "гроза как пейзаж, кадр неба целиком, декоративная вспышка без физического смысла",
    }


def main() -> int:
    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms_in = cell.get("atoms") or cell.get("атомы") or []
    sizes = [float(a.get("size") or 0.11) for a in atoms_in]
    size_med = float(np.median(sizes))

    flash = detect_flash_t(MOL_VID) if os.path.isfile(MOL_VID) else {"t_flash": None, "approx": True}

    # кресты: в клетке 0 — строим для пакета (эталон json не трогаем)
    crosses_raw = cell.get("crosses") or []
    if not crosses_raw:
        crosses_raw = построить_кресты(atoms_in)
    решетка = сводка_решетки(atoms_in, crosses_raw)

    full = []
    adj = defaultdict(list)
    crosses = []
    for i, c in enumerate(crosses_raw):
        ia, ib = int(c["atom_a"]), int(c["atom_b"])
        id_a, id_b = f"grom_full_{ia:05d}", f"grom_full_{ib:05d}"
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
        img = образ_причины(a, flash, size_med)
        links = adj.get(i) or []
        full.append({
            "id": f"grom_full_{i:05d}",
            "стихия": "гром",
            "birth": a.get("birth"),
            "alignment": "axes_decoupled_flash",
            "t_sec_причина": None,  # delay неизвестен
            "звук_ядро": звук_ядро(a, i),
            "образ_причины": img,
            "кресты": {"n": len(links), "связи": links},
            "обратимость": {
                "звук_в_петле": True,
                "образ_записан_в_атом": True,
                "образ_в_синтезе_визуала": False,
                "закон_delay_approx": True,
                "params_104_на_атоме": a.get("params_104") is not None,
                "кресты_на_атоме": True,
                "note": "гром: звук из клетки; вспышка molniya approx; delay null",
            },
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    package = {
        "дата": date.today().isoformat(),
        "контракт": os.path.relpath(CONTRACT, КОРЕНЬ),
        "клетка": "etalon_grom",
        "n_atoms": len(full),
        "size_med": size_med,
        "событие": "flash_shockwave",
        "flash_meta": flash,
        "alignment": {
            "режим": "axes_decoupled_flash",
            "звук_ось": WAV,
            "вспышка_ось": MOL_VID_REL,
            "delay_s": None,
            "note": "синхронной пары нет — закон shockwave_delay не вычисляется",
        },
        "crosses": crosses,
        "число_связей": len(crosses),
        "решетка": решетка,
        "crosses_построены_для_пакета": not bool(cell.get("crosses")),
        "atoms": full,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False)

    side = {
        "дата": date.today().isoformat(),
        "n_crosses": len(crosses),
        "решетка": решетка,
        "пересобраны": not bool(cell.get("crosses")),
        "crosses": crosses,
    }
    with open(OUT_CROSSES, "w", encoding="utf-8") as f:
        json.dump(side, f, ensure_ascii=False)

    meta = {
        "дата": date.today().isoformat(),
        "n_atoms": len(full),
        "n_crosses": len(crosses),
        "size_med": size_med,
        "событие": "flash_shockwave",
        "delay_s": None,
        "t_flash": flash.get("t_flash"),
        "file": os.path.relpath(OUT_JSON, КОРЕНЬ),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # контракт
    contract = f"""# Контракт: образ в атоме грома (flash_shockwave)

> {date.today().isoformat()} · второй эталон после дождя  
> Звук: `grom_real` / `etalon_grom`. Молния — **не звук**, а момент причины.

## Событие

`flash_shockwave` = вспышка (свет) → ударная волна / раскат (звук) с **переменной** задержкой.

Формула (когда есть синхронная пара): `distance_km = delay_s * 0.343`.

## Честность пилота

`molniya/video_*` и `grom_real.wav` — **разные исходники** каталога.  
Поэтому: `delay_s = null`, `закон.approx = true`. Не подделывать синхрон.

## Поля

См. пакет `выход/атомы_полные_гром/атомы_звук_образ.json`.  
Вспышка-намёк: t_flash≈{flash.get('t_flash')} frame={flash.get('frame_i')} xy={flash.get('xy')}.
"""
    with open(CONTRACT, "w", encoding="utf-8") as f:
        f.write(contract)

    lines = [
        "# Полные атомы гром: звук + flash_shockwave",
        "",
        f"> n={len(full)} · crosses={len(crosses)} · t_flash={flash.get('t_flash')} · delay=null (честный approx)",
        "",
        f"- пакет: `{os.path.relpath(OUT_JSON, КОРЕНЬ)}`",
        f"- контракт: `{os.path.relpath(CONTRACT, КОРЕНЬ)}`",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

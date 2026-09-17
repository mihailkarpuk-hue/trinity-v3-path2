# -*- coding: utf-8 -*-
"""Вшить в образ_причины каждого атома огня: как собран визуал v3.

Обновляет рендер + метод_сборки + t0 под sync_meta (single_live_clip).
Запуск: python3 scripts/вшить_метод_визуала_в_атомы_огонь.py
"""
from __future__ import annotations

import json
import os
from datetime import date

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "атомы_звук_образ.json")
SYNC = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "live_sync", "sync_meta.json")
META = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "meta.json")
ARCH = os.path.join(
    os.path.dirname(КОРЕНЬ),
    "Тринити cursor",
    "ворота",
    "АРХИТЕКТУРА_атома_огонь.md",
)
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "метод_визуала_в_атоме_огонь.md")

МЕТОД = {
    "версия_визуала": "v3_под_живое",
    "дата": date.today().isoformat(),
    "эталон_узнаваемости": "данные/стихии_живые/ogon/video_live_01.mp4",
    "сравнение_E": "выход/причина_огонь/E_сравнение_живое_vs_атомы.html",
    "скрипт": "scripts/визуал_из_атомов_огонь.py",
    "слои": [
        "AURA — мягкое тепло под очагом (не чаша)",
        "BODY — 28 языков: ломаная вверх + турбулентность",
        "EDGE — тёмно-красные края языка",
        "CORE — бело-жёлтое ядро низа/середины",
        "WISP — отрывки на кончике",
        "SPARKLE — искры ∝ birth/amp (треск)",
    ],
    "цвет_BGR_канон": {
        "core": [210, 245, 255],
        "hot": [40, 180, 255],
        "mid": [20, 110, 240],
        "edge": [10, 45, 180],
        "запрет": "зелёный канал не доминирует (анти-«свечка» v2)",
    },
    "масштаб": "якорь = медиана геометрии атомов; языки стянуты к одному очагу; height∝size/amp/v_up",
    "закон_в_рендере": {
        "v_up": "высота языка",
        "amp": "яркость ядра + flash около birth",
        "size": "ширина/высота",
        "birth": "фаза вспышки + момент искры",
        "xy": "намёк позиции с того же клипа (map 480x640→холст)",
    },
    "как_сделал": [
        "1. E сравнение живое vs v2 → FAIL «свечка»",
        "2. Взял признаки живого: рваные языки, белое ядро, красная кромка, крупный очаг",
        "3. Переписал рендер: tongue_path турбулентный + heat_color + bloom",
            "4. Записал метод в образ_причины.рендер каждого атома",
            "5. Remux звук' single_live_clip + страница сравнения",
            "6. Усилил масштаб/турбулентность после повторного сравнения рядом",
        ],
    "запрещено": [
        "landscape_campfire",
        "bowl_firepit",
        "hud_circles",
        "foreign_clip",
        "candle_blob_v2",
        "green_tint",
    ],
}


def main() -> int:
    sync = json.load(open(SYNC, encoding="utf-8")) if os.path.isfile(SYNC) else {}
    t0 = float(sync.get("t0_sec") or 4.0)
    pkg = json.load(open(PKG, encoding="utf-8"))
    n = 0
    for a in pkg["atoms"]:
        birth = float(a.get("birth") or 0.0)
        t_clip = round(t0 + birth, 4)
        a["alignment"] = "single_live_clip"
        a["t0_geometry_offset"] = t0
        a["t_sec_причина"] = t_clip
        a["video_причина"] = sync.get("clip") or "данные/стихии_живые/ogon/video_live_01.mp4"
        img = a.get("образ_причины") or {}
        # sync times
        if "треск" in img:
            img["треск"]["t_clip"] = t_clip
        if "пламя" in img:
            img["пламя"]["t_clip"] = t_clip
            img["пламя"]["video"] = a["video_причина"]
        if "геометрия" in img:
            img["геометрия"]["t_sec"] = t_clip
            img["геометрия"]["t_seg"] = birth
            img["геометрия"]["video"] = a["video_причина"]
        img["рендер"] = {
            "тип": "flame_multilayer_v3",
            "версия": МЕТОД["версия_визуала"],
            "запрещено": МЕТОД["запрещено"],
            "слои": МЕТОД["слои"],
            "цвет_канон": МЕТОД["цвет_BGR_канон"],
            "параметры_из_атома": МЕТОД["закон_в_рендере"],
            "скрипт": МЕТОД["скрипт"],
        }
        img["метод_сборки"] = {
            "дата": МЕТОД["дата"],
            "эталон_узнаваемости": МЕТОД["эталон_узнаваемости"],
            "как_сделал": МЕТОД["как_сделал"],
            "масштаб": МЕТОД["масштаб"],
            "alignment": "single_live_clip",
        }
        a["образ_причины"] = img
        obr = a.get("обратимость") or {}
        obr["образ_в_синтезе_визуала"] = True
        obr["single_live_clip"] = True
        obr["визуал_версия"] = МЕТОД["версия_визуала"]
        a["обратимость"] = obr
        n += 1

    pkg["alignment"] = {
        **(pkg.get("alignment") or {}),
        "режим": "single_live_clip",
        "clip": sync.get("clip"),
        "t0_sec": t0,
        "seg_dur_sec": sync.get("seg_dur_sec"),
        "NCC": sync.get("NCC_segment_vs_clip_window", sync.get("NCC")),
    }
    pkg["визуал"] = {
        "версия": МЕТОД["версия_визуала"],
        "mp4": "выход/атомы_полные_огонь/визуал/огонь_из_атомов.mp4",
        "метод": МЕТОД,
        "дата": date.today().isoformat(),
    }
    pkg["метод_визуала_вшит_в_атомы"] = True
    with open(PKG, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False)

    if os.path.isfile(META):
        meta = json.load(open(META, encoding="utf-8"))
        meta["визуал_версия"] = МЕТОД["версия_визуала"]
        meta["t0_sec"] = t0
        meta["метод_визуала_вшит"] = True
        json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(ARCH), exist_ok=True)
    with open(ARCH, "w", encoding="utf-8") as f:
        f.write(
            f"""# Архитектура атома огня

> {date.today().isoformat()} · single_live_clip · визуал v3

## Буква

`birth` + `звук_ядро(+104)` + `кресты` + `образ_причины` + `обратимость`

## Событие / закон

- событие: `combustion_crackle`
- закон: `buoyancy_rise_crackle` (`v_up`, `energy_rel`, crackle@birth)

## Один клип

- клип: `video_live_01.mp4`
- звук эталона = дорожка клипа (сегмент без удара)
- `t_sec_причина = t0 + birth`

## Образ_причины (что внутри)

| блок | смысл |
|------|--------|
| треск | t на оси звука/клипа |
| пламя | video + xy намёк |
| закон | v_up из size/amp |
| геометрия | t_sec/xy с того же клипа |
| **рендер** | тип `flame_multilayer_v3`, слои, цвет, запреты |
| **метод_сборки** | как сделали v3 по сравнению с живым |

## Как сделали визуал v3

"""
            + "\n".join(f"- {s}" for s in МЕТОД["как_сделал"])
            + f"""

Слои: {", ".join(МЕТОД["слои"])}

Скрипт: `{МЕТОД["скрипт"]}`  
Сравнение: `{МЕТОД["сравнение_E"]}`
"""
        )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(
            f"# Метод визуала вшит в атомы огня\n\n"
            f"> {date.today().isoformat()} · n={n}\n\n"
            f"Поля на каждом атоме: `образ_причины.рендер` + `образ_причины.метод_сборки`.\n\n"
            f"Архитектура: `Тринити cursor/ворота/АРХИТЕКТУРА_атома_огонь.md`\n"
        )

    print(json.dumps({"ok": True, "n_atoms": n, "версия": МЕТОД["версия_визуала"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

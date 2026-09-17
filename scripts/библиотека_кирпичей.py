# -*- coding: utf-8 -*-
"""Библиотека кирпичей-моментов — архив полных атомов (протон+электрон+образ).

Канон: СТРОЕНИЕ_АТОМА_МОМЕНТА.md, ПРАВИЛО_СБОРКИ.md
Хранение: данные/библиотека_кирпичей/<источник>.npz (спектры-протон)
         + манифест.json (протон, электрон, образ_кусочек, клетки, оси).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import добыча_словаря as Д  # noqa
import ядро.словарь_кирпичей as СЛ  # noqa
from ядро.кирпич_момент import ОКНО, SR, в_орган  # noqa
from ядро.строение_атома import (  # noqa
    атом_полон,
    mfcc_дельты_по_ряду,
    обогатить_окно_108,
    собрать_слоты_атома,
)

RMS_ЗВУЧИТ = 0.02

ПРАВИЛО = {
    "версия": 3,
    "иерархия": "атом → клетка → орган → организм",
    "атом": {
        "окно_сэмплов": 1024,
        "окно_мс": 46,
        "слоты": "протон + электрон + образ_кусочек",
        "протон": "спектр mag+phase (npz) + 61 ключ (SPECTRAL/VOCAL/MUSICAL/PERCEPTUAL/mfcc/zcr)",
        "электрон": "47 ключей (TEMPORAL/SPATIAL/MOVEMENT/mfcc_delta/AXES); spatial=н/п при моно",
        "образ_кусочек": "точки живого образа в тот же t/старт; цвет из палитры, НЕ freqToHue",
        "метки": "t, старт",
    },
    "клетка": "связная компонента атомов по крестам",
    "орган": "явление; оси fd/mod → копируются в электрон каждого атома",
    "организм": "мелодия из органов (сумма во времени)",
    "сборка_звука": "из_кирпичей(): overlap-add полного спектра; старт из манифеста",
    "инварианты": "ничего не нормируем/сливаем/срезаем; детерминизм; образ синхронен звуку",
    "канон_файл": "СТРОЕНИЕ_АТОМА_МОМЕНТА.md",
}

ВЫХОД = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "данные", "библиотека_кирпичей",
)


def источники() -> list[str]:
    return СЛ.id_корпуса()


def палитра_образа(c: dict) -> list:
    """Палитра живого образа V3: частота → RGB."""
    jp = os.path.join(Д.КЛЕТКИ, c.get("атомы") or "")
    if not os.path.isfile(jp):
        return []
    with open(jp, encoding="utf-8") as f:
        ats = json.load(f).get("atoms") or []
    пал = [
        [float(a.get("freq") or 20), int(a.get("color_r", 128)),
         int(a.get("color_g", 128)), int(a.get("color_b", 128))]
        for a in ats if a.get("color_r") is not None
    ]
    return sorted(пал, key=lambda p: p[0])


def орган_звука(cid: str, x_cache=None):
    """Звук → ОРГАН с полными атомами (протон/электрон/образ_кусочек)."""
    c = Д._cell(cid)
    ap = Д._audio_path(c)
    if not ap:
        return None
    if x_cache is not None:
        x, sr = x_cache
    else:
        x, sr = Д._load_audio(ap)
        x = np.asarray(x, float)
        if sr != SR:
            x = np.interp(
                np.linspace(0, len(x) - 1, int(len(x) * SR / sr)),
                np.arange(len(x)), x,
            )
            sr = SR
        x = x / (np.max(np.abs(x)) + 1e-12)

    орг = в_орган(x, sr)
    атомы = орг["атомы"]
    оси = орг["орган"]["оси"] or {}
    палитра = палитра_образа(c)

    зв = [i for i, a in enumerate(атомы) if a["параметры"].get("rms", 0) >= RMS_ЗВУЧИТ]
    if not зв:
        return None
    remap = {old: new for new, old in enumerate(зв)}

    # 1) полный 108 на каждое звучащее окно
    params_list = []
    for old in зв:
        a = атомы[old]
        i0 = int(a["старт"])
        seg = x[i0:i0 + ОКНО]
        if len(seg) < ОКНО:
            seg = np.pad(seg, (0, ОКНО - len(seg)))
        p108 = обогатить_окно_108(seg, sr)
        # мгновенные (hps и др.) не затираем, если полезнее
        for k, v in a["параметры"].items():
            if k not in p108 or (k.endswith("_hps") and v):
                p108[k] = float(v)
        # оси органа → электрон каждого атома (поле, в котором живёт момент)
        for k, v in оси.items():
            p108[k] = float(v)
        params_list.append(p108)

    # 2) mfcc-дельты вдоль ряда звучащих атомов
    дельты = mfcc_дельты_по_ряду(params_list)
    for p, d in zip(params_list, дельты):
        p.update(d)

    A = []
    n_полных = 0
    for j, old in enumerate(зв):
        a = атомы[old]
        f = dict(a["форма"])
        кресты = sorted(remap[k] for k in f.get("кресты", []) if k in remap)
        f_clean = {
            kk: (round(float(v), 4) if isinstance(v, (int, float)) else v)
            for kk, v in f.items() if kk != "кресты"
        }
        пики = sorted(a.get("пики", []), key=lambda p: -p["амплитуда"])[:12]
        пики_c = [[round(p["частота"], 1), round(p["амплитуда"], 4)] for p in пики]

        слоты = собрать_слоты_атома(
            t=a["t"], старт=int(a["старт"]),
            params_108=params_list[j],
            пики=пики, форма=f_clean,
            палитра=палитра, источник=cid, mono=True,
        )
        rec = {
            "t": round(float(a["t"]), 4),
            "старт": int(a["старт"]),
            "форма": f_clean,
            "пики": пики_c,
            "кресты": кресты,
            # совместимость + канон слотов:
            "параметры": {kk: round(float(v), 5) for kk, v in a["параметры"].items()},
            "протон": слоты["протон"],
            "электрон": слоты["электрон"],
            "образ_кусочек": слоты["образ_кусочек"],
            "spatial_статус": слоты["spatial_статус"],
            "полнота": слоты["полнота"],
            "mag": a["спектр_магнитуда"],
            "phase": a["спектр_фаза"],
        }
        if атом_полон(rec):
            n_полных += 1
        A.append(rec)

    клетки = [sorted(remap[i] for i in cl if i in remap) for cl in орг["клетки"]]
    клетки = [cl for cl in клетки if cl]
    return {
        "атомы": A,
        "клетки": клетки,
        "оси": оси,
        "палитра": палитра,
        "длит": len(x) / sr,
        "атомов_полных": n_полных,
    }


def main() -> None:
    только = sys.argv[1:] or None
    ids = источники()
    if только:
        ids = [c for c in ids if c in только]
    os.makedirs(ВЫХОД, exist_ok=True)
    органы = []
    всего_атомов = 0
    всего_полных = 0
    печать = []
    for cid in ids:
        try:
            о = орган_звука(cid)
        except Exception as e:  # noqa
            print(f"✗ {cid}: {e}")
            continue
        if not о:
            continue
        A = о["атомы"]
        mags = np.array([a["mag"] for a in A], dtype=np.float32)
        phs = np.array([a["phase"] for a in A], dtype=np.float32)
        np.savez_compressed(os.path.join(ВЫХОД, cid + ".npz"), mag=mags, phase=phs)
        органы.append({
            "источник": cid,
            "атомов": len(A),
            "атомов_полных": о["атомов_полных"],
            "клеток": len(о["клетки"]),
            "оси": о["оси"],
            "длительность": round(о["длит"], 4),
            "палитра_образа": о["палитра"],
            "клетки": о["клетки"],
            "атомы": [
                {
                    "индекс": j,
                    "t": round(a["t"], 4),
                    "старт": a["старт"],
                    "форма": a["форма"],
                    "пики": a["пики"],
                    "кресты": a["кресты"],
                    "параметры": a["параметры"],
                    "протон": a["протон"],
                    "электрон": a["электрон"],
                    "образ_кусочек": a["образ_кусочек"],
                    "spatial_статус": a["spatial_статус"],
                    "полнота": a["полнота"],
                }
                for j, a in enumerate(A)
            ],
        })
        всего_атомов += len(A)
        всего_полных += о["атомов_полных"]
        печать.append((cid, len(A), о["атомов_полных"], len(о["клетки"])))
        print(f"  {cid}: {len(A)} атомов, полных {о['атомов_полных']}")

    with open(os.path.join(ВЫХОД, "манифест.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "атомов": всего_атомов,
                "атомов_полных": всего_полных,
                "органов": len(органы),
                "окно": ОКНО,
                "шаг": ОКНО // 2,
                "sr": SR,
                "иерархия": "атом → клетка → орган → организм",
                "строение_атома": "протон + электрон + образ_кусочек",
                "правило_сборки": ПРАВИЛО,
                "органы": органы,
            },
            f,
            ensure_ascii=False,
        )
    печать.sort(key=lambda x: -x[1])
    print(f"библиотека: {всего_атомов} атомов ({всего_полных} полных) "
          f"в {len(органы)} органах → {ВЫХОД}")
    for row in печать[:5] + [("…", 0, 0, 0)] + печать[-3:]:
        print(f"  {row[0]:22s} {row[1]} ат / полных {row[2]} / {row[3]} клеток")


if __name__ == "__main__":
    main()

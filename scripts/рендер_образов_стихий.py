# -*- coding: utf-8 -*-
"""Рендер чистых природных образов из образ_кусочек архива.

Не freqToHue: цвет уже в точках из палитры живого образа.
X = время (поз_x), Y = log-частота (y в кусочке).

Запуск:
  python3 scripts/рендер_образов_стихий.py
  python3 scripts/рендер_образов_стихий.py etalon_dozhd etalon_veter
Выход: выход/образы_стихий/<id>.png + index.html
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "-q"])
    from PIL import Image, ImageDraw

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
МАН = os.path.join(КОРЕНЬ, "данные", "библиотека_кирпичей", "манифест.json")
ВЫХОД = os.path.join(КОРЕНЬ, "выход", "образы_стихий")

# стихии по запросу автора (реки нет в корпусе → водопад как «вода»)
СТИХИИ = [
    ("etalon_dozhd", "Дождь"),
    ("etalon_veter", "Ветер"),
    ("etalon_ogon", "Огонь"),
    ("etalon_vodopad", "Вода (водопад)"),
]

W, H = 1200, 720
PAD = 48


def _орган(man: dict, cid: str) -> dict | None:
    for o in man["органы"]:
        if o["источник"] == cid:
            return o
    return None


def _фон_для(имя: str) -> tuple[int, int, int]:
    # атмосфера, не плоский серый
    return {
        "Дождь": (12, 22, 32),
        "Ветер": (28, 34, 40),
        "Огонь": (28, 12, 8),
        "Вода (водопад)": (8, 24, 28),
    }.get(имя, (16, 18, 20))


def рендер(орган: dict, title: str, path: str) -> dict:
    pts = []
    for a in орган["атомы"]:
        img = a.get("образ_кусочек") or {}
        for t in img.get("точки") or []:
            pts.append(t)
    bg = _фон_для(title)
    im = Image.new("RGB", (W, H), bg)
    dr = ImageDraw.Draw(im, "RGBA")

    if not pts:
        im.save(path)
        return {"точек": 0, "файл": path}

    xs = [float(p.get("x") or 0) for p in pts]
    ys = [float(p.get("y") or 0) for p in pts]
    # лёгкий джиттер не нужен — детерминизм
    for p in pts:
        x = PAD + float(p.get("x") or 0) * (W - 2 * PAD)
        y = PAD + (1.0 - float(p.get("y") or 0)) * (H - 2 * PAD)  # верх = высокие частоты
        amp = float(p.get("размер") or 0.05)
        r = max(2, min(18, int(3 + amp * 40)))
        col = (int(p.get("r", 128)), int(p.get("g", 128)), int(p.get("b", 128)), 200)
        dr.ellipse([x - r, y - r, x + r, y + r], fill=col)

    # подпись
    dr.text((PAD, H - 28), f"{title}  ·  {орган['источник']}  ·  {len(pts)} точек", fill=(200, 200, 200))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    return {
        "точек": len(pts),
        "атомов": орган["атомов"],
        "палитра": len(орган.get("палитра_образа") or []),
        "файл": path,
        "x_range": [round(min(xs), 3), round(max(xs), 3)],
        "y_range": [round(min(ys), 3), round(max(ys), 3)],
    }


def html(карточки: list) -> str:
    cards = []
    for c in карточки:
        rel = os.path.basename(c["файл"])
        cards.append(
            f'<figure><img src="{rel}" alt="{c["имя"]}" loading="lazy">'
            f'<figcaption>{c["имя"]} · {c["точек"]} точек · палитра {c["палитра"]}</figcaption></figure>'
        )
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Образы стихий</title>
<style>
body{{margin:0;background:#0a0e10;color:#e8eeea;font:16px/1.4 Georgia,serif}}
main{{max-width:1100px;margin:0 auto;padding:3rem 1.5rem}}
h1{{font-weight:500;font-size:2rem;margin:0 0 .4rem}}
p{{color:#9aa8a0;max-width:36rem}}
.grid{{display:grid;gap:2rem;margin-top:2rem}}
figure{{margin:0}}
img{{width:100%;height:auto;display:block}}
figcaption{{margin-top:.5rem;color:#9aa8a0;font-size:.9rem}}
.note{{margin-top:2rem;font-size:.85rem;color:#6a7870}}
</style></head><body><main>
<h1>Образы стихий</h1>
<p>Из архива атомов: образ_кусочек, цвет палитры живого образа — не freqToHue.
Реки в корпусе нет; «вода» = водопад.</p>
<div class="grid">{"".join(cards)}</div>
<p class="note">выход/образы_стихий · scripts/рендер_образов_стихий.py</p>
</main></body></html>
"""


def main() -> None:
    with open(МАН, encoding="utf-8") as f:
        man = json.load(f)
    want = sys.argv[1:]
    items = СТИХИИ
    if want:
        items = [(cid, cid) for cid in want]

    os.makedirs(ВЫХОД, exist_ok=True)
    cards = []
    print("образы стихий →", ВЫХОД)
    for cid, имя in items:
        о = _орган(man, cid)
        if not о:
            print(f"  ✗ нет органа {cid}")
            continue
        path = os.path.join(ВЫХОД, f"{cid}.png")
        info = рендер(о, имя, path)
        print(f"  {имя:20s} {info['точек']:5d} точек  pal={info['палитра']}  → {path}")
        cards.append({"имя": имя, "файл": path, **info})

    index = os.path.join(ВЫХОД, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(html(cards))
    print("галерея:", index)

    # поправить отбор: стихии — первая очередь
    otch = os.path.join(КОРЕНЬ, "отчёты", "отбор_чистые_образы.md")
    with open(otch, "w", encoding="utf-8") as f:
        f.write(
            """# Чистые образы — стихии (решение автора)

Буквы по метрике «тональность» набирали score, но **чистый живой образ** —
это стихия, не буква. Начинаем с природы.

## Первая очередь

| Звук | id | В корпусе |
|------|-----|-----------|
| Дождь | `etalon_dozhd` | да |
| Ветер | `etalon_veter` | да |
| Огонь | `etalon_ogon` | да |
| Вода | `etalon_vodopad` | да (водопад; отдельной «реки» нет) |
| Река | — | **нет в корпусе** — добавить запись или взять водопад как прокси |

Рендер: `выход/образы_стихий/` · `python3 scripts/рендер_образов_стихий.py`
"""
        )


if __name__ == "__main__":
    main()

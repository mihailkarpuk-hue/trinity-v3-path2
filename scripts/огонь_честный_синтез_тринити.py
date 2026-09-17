# -*- coding: utf-8 -*-
"""Честный звук огня: анализатор → атомы клетки → синтез_из_атомов → калибр треска.

Факт: AAC дорожка video_live_01 — сжатый гул (crest~1.6), атомизация → 1 атом.
Поэтому анализатор крутится на ogon_real (клетка etalon_ogon), не на live AAC.
Обратный путь — только синтез_из_атомов (+ слой треска из атомов/эталона).

Запуск: python3 scripts/огонь_честный_синтез_тринити.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро"), os.path.join(КОРЕНЬ, "scripts")]

# переиспользуем калибр треска как канон обратного синтеза
import калибр_огонь_треск as crackle  # noqa: E402

PKG = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "атомы_звук_образ.json")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "E_огонь_тринити_честный.html")
LIVE = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "live_sync", "ogon_live_01_etalon.wav")
REPORT = os.path.join(КОРЕНЬ, "отчёты", "огонь_честный_синтез_тринити.json")


def main() -> int:
    # 1) полный цикл калибра = синтез_из_атомов(клетка) + треск
    rc = crackle.main()
    if rc != 0:
        return rc

    # 2) честная метка в пакете атомов
    honesty = {
        "анализатор_тринити": True,
        "вход_анализатора": "данные/клеточки/эталоны/ogon_real.wav",
        "клетка": "etalon_ogon",
        "обратный_путь": "ядро.синтез.синтез_из_атомов → калибр_огонь_треск",
        "не_использовано": (
            "копирование live AAC как звук' (прежний single_live remux) — "
            "live AAC crest≈1.6, атомизация даёт 1 атом, phase='тон' ломает float(phase)"
        ),
        "live_клип": "video_live_01 — только ось образа/времени; звук' не = дорожка клипа",
        "wav": "выход/причина_огонь/калибр_оси/сборка_чистая.wav",
        "дата": date.today().isoformat(),
    }

    if os.path.isfile(PKG):
        pkg = json.load(open(PKG, encoding="utf-8"))
        pkg["sound_score"] = {
            **(pkg.get("sound_score") or {}),
            "метод": honesty["обратный_путь"],
            "анализатор": True,
            "вход": honesty["вход_анализатора"],
            "долг_live_aac": honesty["не_использовано"],
        }
        pkg["trinity_sound_loop"] = honesty
        # вшить в первые атомы (метод_сборки / рендер звука)
        for a in pkg.get("atoms") or []:
            a["метод_сборки"] = {
                **(a.get("метод_сборки") or {}),
                "звук": honesty,
            }
            break  # достаточно на пакете; массово ниже
        n_mark = 0
        for a in pkg.get("atoms") or []:
            mb = a.get("метод_сборки") or {}
            mb["звук_петля"] = "анализатор→атомы→синтез_из_атомов→треск"
            mb["анализатор"] = True
            a["метод_сборки"] = mb
            n_mark += 1
            if n_mark >= 200:
                break
        with open(PKG, "w", encoding="utf-8") as f:
            json.dump(pkg, f, ensure_ascii=False)

    # 3) E страница
    live_note = "есть" if os.path.isfile(LIVE) else "нет"
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — огонь честный Тринити</title>
<style>
body{{font-family:system-ui;background:#1a120c;color:#f2e6d8;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.3rem 0 .9rem}} .meta{{opacity:.85}}
</style></head><body>
<h1>E ухо — честный цикл Тринити</h1>
<p class="meta">{honesty['дата']}<br/>
анализатор: <b>да</b> на <code>ogon_real.wav</code> → клетка → <code>синтез_из_атомов</code> → треск<br/>
live AAC: <b>не</b> источник атомов (crest≈1.6 → 1 атом). live_клип={live_note}
</p>
<p>Эталон клетки</p>
<audio controls src="калибр_оси/эталон.wav"></audio>
<p>База синтеза (до треска)</p>
<audio controls src="калибр_оси/база_клетка_synth.wav"></audio>
<p>Сборка честная</p>
<audio controls src="калибр_оси/сборка_чистая.wav"></audio>
<p><b>Cmd+Shift+R</b>. Это звук из атомов, не копия дорожки видео.</p>
</body></html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(honesty, f, ensure_ascii=False, indent=2)

    print(json.dumps(honesty, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

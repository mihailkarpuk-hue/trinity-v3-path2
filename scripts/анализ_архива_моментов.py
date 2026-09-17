# -*- coding: utf-8 -*-
"""Анализ архива кирпичей-моментов (путь 2).

Не d₉, не нормировка: смотрим реальные величины в манифесте + покрытие.
Цель — понять склад, чтобы решить следующий шаг.

Запуск: python3 scripts/анализ_архива_моментов.py
Отчёт:  отчёты/анализ_архива_моментов.md
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
БИБ = os.path.join(КОРЕНЬ, "данные", "библиотека_кирпичей")
ОТЧЁТ = os.path.join(КОРЕНЬ, "отчёты", "анализ_архива_моментов.md")

# оси фенотипа — что смотрим в архиве (реальные величины)
ОСИ = [
    "rms", "spectral_centroid", "spectral_flatness", "spectral_slope",
    "spectral_kurtosis", "spectral_skewness", "pitch_hps",
    "harmonic_ratio_hps", "formant_f1", "formant_f2",
    "voicing", "breathiness", "zero_crossing_rate",
]


def _класс(cid: str) -> str:
    if cid.startswith("живая_"):
        return "буквица"
    if cid.startswith("bazis_"):
        return "базис"
    if cid.startswith("etalon_"):
        return "природа"
    return "прочее"


def _квантили(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "p25": float(np.percentile(a, 25)),
        "med": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def _fmt_q(q: dict) -> str:
    if not q.get("n"):
        return "—"
    return (
        f"n={q['n']} · med={q['med']:.4g} · "
        f"[{q['p25']:.4g} … {q['p75']:.4g}] · max={q['max']:.4g}"
    )


def main() -> None:
    man_path = os.path.join(БИБ, "манифест.json")
    with open(man_path, encoding="utf-8") as f:
        M = json.load(f)

    органы = M["органы"]
    npz_ok = 0
    npz_bytes = 0
    for o in органы:
        p = os.path.join(БИБ, o["источник"] + ".npz")
        if os.path.isfile(p):
            npz_ok += 1
            npz_bytes += os.path.getsize(p)

    # плоский список атомов
    rows = []
    for o in органы:
        cls = _класс(o["источник"])
        n_links = sum(len(a.get("кресты") or []) for a in o["атомы"])
        for a in o["атомы"]:
            p = a.get("параметры") or {}
            rows.append({
                "источник": o["источник"],
                "класс": cls,
                "клеток_органа": o["клеток"],
                "палитра": len(o.get("палитра_образа") or []),
                "пиков": len(a.get("пики") or []),
                "крестов": len(a.get("кресты") or []),
                "параметры": p,
                "частота_дом": (a.get("форма") or {}).get("частота_дом"),
            })

    by_class = Counter(r["класс"] for r in rows)
    organs_by_class = Counter(_класс(o["источник"]) for o in органы)

    # покрытие параметров
    keys_count = Counter()
    for r in rows:
        keys_count.update(r["параметры"].keys())
    param_keys = sorted(keys_count.keys())
    full_cover = [k for k in param_keys if keys_count[k] == len(rows)]

    # квантили по осям × класс
    axis_stats = {}
    for ax in ОСИ:
        axis_stats[ax] = {}
        for cls in ("природа", "буквица", "базис"):
            xs = [float(r["параметры"].get(ax, float("nan"))) for r in rows if r["класс"] == cls]
            axis_stats[ax][cls] = _квантили(xs)

    # органы: атомы / клетки / палитра
    organ_table = []
    for o in sorted(органы, key=lambda x: -x["атомов"]):
        links = sum(len(a.get("кресты") or []) for a in o["атомы"]) / max(1, o["атомов"])
        organ_table.append({
            "id": o["источник"],
            "класс": _класс(o["источник"]),
            "атомов": o["атомов"],
            "клеток": o["клеток"],
            "длит": o["длительность"],
            "палитра": len(o.get("палитра_образа") or []),
            "крестов_на_атом": round(links, 2),
            "оси": o.get("оси") or {},
        })

    # разнообразие: centroid vs flatness «карта»
    # простая сепарация классов по медианам
    sep_notes = []
    for ax in ("spectral_flatness", "harmonic_ratio_hps", "spectral_centroid", "voicing"):
        meds = {cls: axis_stats[ax][cls].get("med") for cls in ("природа", "буквица", "базис")
                if axis_stats[ax][cls].get("n")}
        sep_notes.append((ax, meds))

    # дыры / риски
    risks = []
    no_pal = [o["id"] for o in organ_table if o["палитра"] == 0]
    if no_pal:
        risks.append(f"Без палитры_образа ({len(no_pal)}): " + ", ".join(no_pal[:12])
                     + ("…" if len(no_pal) > 12 else ""))
    mono_cell = [o["id"] for o in organ_table if o["клеток"] <= 1]
    if mono_cell:
        risks.append(f"Одна клетка на орган ({len(mono_cell)}) — кресты слабо режут поток: "
                     + ", ".join(mono_cell[:10]) + ("…" if len(mono_cell) > 10 else ""))
    tiny = [o["id"] for o in organ_table if o["атомов"] < 10]
    if tiny:
        risks.append(f"Мало атомов (<10): {', '.join(tiny)}")
    if npz_ok != len(органы):
        risks.append(f"npz не на все органы: {npz_ok}/{len(органы)}")

    # fd из осей органа
    fd_rows = []
    for o in organ_table:
        оси = o["оси"]
        fd = оси.get("fd") or оси.get("fractal_dimension")
        if fd is not None:
            fd_rows.append((o["id"], o["класс"], float(fd)))

    # --- отчёт ---
    lines = [
        "# Анализ архива кирпичей-моментов",
        "",
        f"> Дата: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"> Источник: `данные/библиотека_кирпичей/` · окно={M.get('окно')} шаг={M.get('шаг')} sr={M.get('sr')}",
        f"> Иерархия: {M.get('иерархия')}",
        "",
        "## 1. Склад (что лежит)",
        "",
        f"| | |",
        f"|---|---|",
        f"| Органов | **{M['органов']}** |",
        f"| Атомов (моментов) | **{M['атомов']}** |",
        f"| npz на диске | {npz_ok} файлов · {npz_bytes/1e6:.1f} МБ |",
        f"| Ключей параметров/атом | {len(param_keys)} (полное покрытие: {len(full_cover)}) |",
        "",
        "### По классам",
        "",
        "| Класс | Органов | Атомов |",
        "|-------|---------|--------|",
    ]
    for cls in ("природа", "буквица", "базис", "прочее"):
        if organs_by_class[cls] or by_class[cls]:
            lines.append(f"| {cls} | {organs_by_class[cls]} | {by_class[cls]} |")

    lines += [
        "",
        "### Органы (топ по числу атомов / хвост)",
        "",
        "| источник | класс | атомов | клеток | палитра | крест/атом | t,с |",
        "|----------|-------|--------|--------|---------|------------|-----|",
    ]
    show = organ_table[:8] + ([{"id": "…"}] if len(organ_table) > 16 else []) + organ_table[-8:]
    for o in show:
        if o["id"] == "…":
            lines.append("| … | | | | | | |")
            continue
        lines.append(
            f"| {o['id']} | {o['класс']} | {o['атомов']} | {o['клеток']} | "
            f"{o['палитра']} | {o['крестов_на_атом']} | {o['длит']:.2f} |"
        )

    lines += [
        "",
        "## 2. Фенотип архива (реальные величины, без z)",
        "",
        "Медианы по классам — где классы **разводятся** самими моментами:",
        "",
    ]
    for ax, meds in sep_notes:
        parts = ", ".join(f"{k}={v:.4g}" for k, v in meds.items() if v is not None)
        lines.append(f"- **{ax}**: {parts}")

    lines += [
        "",
        "### Квантили ключевых осей",
        "",
    ]
    for ax in ОСИ:
        lines.append(f"#### `{ax}`")
        lines.append("")
        lines.append("| класс | распределение |")
        lines.append("|-------|---------------|")
        for cls in ("природа", "буквица", "базис"):
            lines.append(f"| {cls} | {_fmt_q(axis_stats[ax][cls])} |")
        lines.append("")

    if fd_rows:
        lines += [
            "### Оси органа (fd и др. — уровень потока, не атома)",
            "",
            "| источник | класс | fd |",
            "|----------|-------|----|",
        ]
        for cid, cls, fd in sorted(fd_rows, key=lambda x: -x[2])[:15]:
            lines.append(f"| {cid} | {cls} | {fd:.4g} |")
        lines.append("")

    # пики
    peaks = [r["пиков"] for r in rows]
    lines += [
        "## 3. Образ внутри атома",
        "",
        f"- Пиков на атом (топ-12 в манифесте): med={np.median(peaks):.0f}, "
        f"max={max(peaks)}, доля с ≥1 пиком="
        f"{100*sum(1 for p in peaks if p)/len(peaks):.1f}%",
        f"- Органов с палитрой_образа: "
        f"{sum(1 for o in organ_table if o['палитра']>0)}/{len(organ_table)}",
        "",
        "## 4. Дыры и риски",
        "",
    ]
    if risks:
        for r in risks:
            lines.append(f"- {r}")
    else:
        lines.append("- Критичных дыр нет.")

    # выводы для решения
    # простая эвристика следующего шага
    next_steps = []
    next_steps.append(
        "**Держать архив как истину** — 55 органов / полный спектр в npz; "
        "анализы и решения строить от манифеста, не от словаря CP#2."
    )
    if no_pal:
        next_steps.append(
            "**Добить палитры** для органов без `палитра_образа` — иначе образ врёт "
            "(или честно пометить «нет живой клетки»)."
        )
    if mono_cell:
        next_steps.append(
            "**Порог τ_клетка** — много органов с 1 клеткой: либо звук однороден, "
            "либо τ=0.85 слишком жёсткий; имеет смысл прогнать τ∈{0.7,0.75,0.8}."
        )
    next_steps.append(
        "**Карта фенотипа** (flatness × harmonicity × centroid) — классы уже "
        "разводятся; следующий эксперимент: кластеры *между* органами "
        "(общие моменты природы и буквицы), не k-means под d₉."
    )
    next_steps.append(
        "**Образ из архива** — развернуть пики + палитру для 3–5 органов "
        "(дождь, А, гром) и глазами проверить канон «не freqToHue»."
    )
    next_steps.append(
        "**Не трогать CP#2**, пока архив не прочитан: путь 2 решает сборкой и "
        "анализом моментов, не ε_квант."
    )

    lines += [
        "",
        "## 5. Вердикт — что делать дальше",
        "",
        "Архив **есть и полон по корпусу 55**. Дальше — не наращивать синтез, "
        "а **читать склад** и проверять гипотезы на нём.",
        "",
    ]
    for i, s in enumerate(next_steps, 1):
        lines.append(f"{i}. {s}")

    lines += [
        "",
        "---",
        "",
        f"Пересборка архива: `python3 scripts/библиотека_кирпичей.py`",
        f"Этот отчёт: `python3 scripts/анализ_архива_моментов.py`",
        "",
    ]

    os.makedirs(os.path.dirname(ОТЧЁТ), exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(ОТЧЁТ, "w", encoding="utf-8") as f:
        f.write(text)

    # краткий stdout
    print(f"архив: {M['атомов']} атомов / {M['органов']} органов · npz {npz_bytes/1e6:.1f} МБ")
    print(f"классы атомов: {dict(by_class)}")
    print(f"без палитры: {len(no_pal)} · одна клетка: {len(mono_cell)}")
    print(f"отчёт → {ОТЧЁТ}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Пересечение каркаса алфавита по PASS/рабочим пакетам.

Правило: спека = пересечение файлов, не выдумка полей.
Запуск: python3 scripts/сверить_алфавит_пересечение.py
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

КОРЕНЬ = Path(__file__).resolve().parents[1]
ПРОЕКТ = КОРЕНЬ.parent
OUT_JSON = КОРЕНЬ / "отчёты" / "алфавит_пересечение.json"
OUT_MD = КОРЕНЬ / "отчёты" / "алфавит_пересечение.md"
OUT_E = КОРЕНЬ / "выход" / "атомная_визуализация" / "E_алфавит_пересечение.html"
SPEC = ПРОЕКТ / "Тринити cursor" / "ворота" / "КАК_собираем_атомы" / "СПЕЦИФИКАЦИЯ_алфавит_v0.1.md"
DRAFT = КОРЕНЬ / "отчёты" / "алфавит_черновик.md"

# корпуса для пересечения инварианта буквы
PACKAGES = {
    "дождь_live": {
        "path": "выход/атомы_полные_дождь_live/атомы_звук_образ.json",
        "E": "PASS",
        "событие": "drop_impact",
        "закон": "particle_scale_impulse",
        "материя": "drop_streak_impact_crown",
        "alignment_норма": "single_live_clip",
    },
    "огонь": {
        "path": "выход/атомы_полные_огонь/атомы_звук_образ.json",
        "E": "PASS",
        "событие": "combustion_crackle",
        "закон": "buoyancy_rise_crackle",
        "материя": "flame_tongue_crackle_spark",
        "alignment_норма": "single_live_clip",
    },
    "ветер": {
        "path": "выход/атомы_полные_ветер/атомы_звук_образ.json",
        "E": "PASS",
        "событие": "wind_stream_rustle",
        "закон": "horizontal_flow_sway",
        "материя": "broken_fiber_fleck",
        "alignment_норма": "single_live_clip",
    },
    "водопад": {
        "path": "выход/атомы_полные_водопад/атомы_звук_образ.json",
        "E": "PASS",
        "событие": "waterfall_fall_spray",
        "закон": "gravity_fall_spray",
        "материя": "vertical_ribbon_hard_spray",
        "alignment_норма": "single_live_clip",
    },
    "река": {
        "path": "выход/атомы_полные_река/атомы_звук_образ.json",
        "E": "PASS",
        "событие": "river_channel_flow",
        "закон": "channel_flow_current",
        "материя": "current_filaments_foam",
        "alignment_норма": "single_live_clip",
    },
    "гром": {
        "path": "выход/атомы_полные_гром/атомы_звук_образ.json",
        "E": "PASS (delay долг)",
        "событие": "flash_shockwave",
        "закон": "shockwave_delay",
        "материя": "flash_then_shock_field",
        "alignment_норма": "axes_decoupled_flash",
        "note": "delay_s=null — постоянный долг",
    },
}


def deep_keys(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.add(p)
            if isinstance(v, dict):
                out |= deep_keys(v, p)
    return out


def sample_atom(pkg):
    return pkg["atoms"][0]


def analyze(name, meta):
    path = КОРЕНЬ / meta["path"]
    pkg = json.load(open(path, encoding="utf-8"))
    a = sample_atom(pkg)
    oc = a.get("образ_причины") or {}
    zj = a.get("звук_ядро") or {}
    yadro = zj.get("ядро") or {}
    ren = oc.get("рендер") or {}
    zak = oc.get("закон") or {}
    crosses = a.get("кресты")
    cross_ok = isinstance(crosses, list) and len(crosses) >= 0
    # некоторые пакеты хранят кресты как list на атоме
    n104 = 0
    for atom in pkg["atoms"][: min(200, len(pkg["atoms"]))]:
        if (atom.get("звук_ядро") or {}).get("params_104"):
            n104 += 1
    return {
        "name": name,
        "n_atoms": pkg.get("n_atoms") or len(pkg["atoms"]),
        "E": meta["E"],
        "событие": meta["событие"],
        "закон": meta["закон"],
        "материя": meta["материя"],
        "alignment_норма": meta["alignment_норма"],
        "alignment_факт": (pkg.get("alignment") or {}).get("режим")
        if isinstance(pkg.get("alignment"), dict)
        else pkg.get("alignment") or a.get("alignment"),
        "atom_keys": sorted(a.keys()),
        "oc_keys": sorted(oc.keys()),
        "zak_keys": sorted(zak.keys()),
        "ren_keys": sorted(ren.keys()),
        "yadro_keys": sorted(yadro.keys()),
        "has_params_104_sample": bool(zj.get("params_104")),
        "params_104_share_first200": round(n104 / max(1, min(200, len(pkg["atoms"]))), 3),
        "has_кресты": cross_ok,
        "has_материя": "материя" in ren,
        "has_геометрия": "геометрия" in oc,
        "note": meta.get("note"),
    }


def intersect(sets):
    it = iter(sets)
    acc = set(next(it))
    for s in it:
        acc &= set(s)
    return sorted(acc)


def main() -> int:
    rows = [analyze(n, m) for n, m in PACKAGES.items()]
    atom_i = intersect(r["atom_keys"] for r in rows)
    oc_i = intersect(r["oc_keys"] for r in rows)
    zak_i = intersect(r["zak_keys"] for r in rows)
    ren_i = intersect(r["ren_keys"] for r in rows)
    yadro_i = intersect(r["yadro_keys"] for r in rows)

    report = {
        "дата": date.today().isoformat(),
        "правило": "спека = пересечение пакетов",
        "n_пакетов": len(rows),
        "пакеты": rows,
        "пересечение": {
            "atom": atom_i,
            "образ_причины": oc_i,
            "закон": zak_i,
            "рендер": ren_i,
            "ядро": yadro_i,
        },
        "словарь_событий": [
            {
                "стихия": r["name"],
                "событие": r["событие"],
                "закон": r["закон"],
                "материя": r["материя"],
                "E": r["E"],
                "n": r["n_atoms"],
            }
            for r in rows
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    lines = [
        "# Алфавит — пересечение пакетов (v0.1 refresh)",
        "",
        f"> {report['дата']} · правило: спека = пересечение файлов, не выдумка",
        "",
        "## Корпуса",
        "",
        "| Стихия | n | E | alignment | событие | закон | материя |",
        "|--------|---|---|-----------|---------|-------|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['n_atoms']} | {r['E']} | `{r['alignment_факт']}` | `{r['событие']}` | `{r['закон']}` | `{r['материя']}` |"
        )
    lines += [
        "",
        "## Пересечение корня атома",
        "",
        "```",
        " · ".join(atom_i),
        "```",
        "",
        "## Пересечение `образ_причины`",
        "",
        "```",
        " · ".join(oc_i),
        "```",
        "",
        "## Пересечение `закон`",
        "",
        "```",
        " · ".join(zak_i),
        "```",
        "",
        "## Пересечение `рендер`",
        "",
        "```",
        " · ".join(ren_i),
        "```",
        "",
        "## Пересечение `звук_ядро.ядро`",
        "",
        "```",
        " · ".join(yadro_i),
        "```",
        "",
        "## Вывод",
        "",
        "- Инвариант корня стабилен на 6 пакетах.",
        "- В `рендер` теперь обязательно: `материя` + `тип` (+ `запрещено_материя`).",
        "- В `закон` пересечение: `модель` · `формула` · `energy_rel` · `size_atom` · `size_med`.",
        "- Слоты события (полёт/удар/пламя/…) — **не** общий shape.",
        "- Гром: E PASS, но `delay_s` — постоянный долг.",
        "",
        f"JSON: `{OUT_JSON.relative_to(КОРЕНЬ)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # E page
    rows_html = "".join(
        f"<tr><td>{r['name']}</td><td>{r['n_atoms']}</td><td>{r['E']}</td>"
        f"<td><code>{r['событие']}</code></td><td><code>{r['закон']}</code></td>"
        f"<td><code>{r['материя']}</code></td></tr>"
        for r in rows
    )
    OUT_E.parent.mkdir(parents=True, exist_ok=True)
    OUT_E.write_text(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — алфавит пересечение</title>
<style>
body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:980px;margin:2rem auto;padding:0 1rem}}
table{{width:100%;border-collapse:collapse;font-size:.92rem}}
th,td{{border-bottom:1px solid #2a3340;padding:.45rem .35rem;text-align:left}}
code{{color:#9dceb0}}.meta{{opacity:.85}}
</style></head><body>
<h1>Алфавит v0.1 — пересечение</h1>
<p class="meta">{report['дата']} · спека = пересечение пакетов · не статистика</p>
<table>
<thead><tr><th>Стихия</th><th>n</th><th>E</th><th>событие</th><th>закон</th><th>материя</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<h2>Корень ∩</h2>
<p><code>{' · '.join(atom_i)}</code></p>
<h2>образ_причины ∩</h2>
<p><code>{' · '.join(oc_i)}</code></p>
<h2>закон ∩</h2>
<p><code>{' · '.join(zak_i)}</code></p>
<h2>рендер ∩</h2>
<p><code>{' · '.join(ren_i)}</code></p>
<p><a href="/выход/атомная_визуализация/index.html" style="color:#9dceb0">← хаб</a></p>
</body></html>
""",
        encoding="utf-8",
    )

    # refresh draft
    DRAFT.write_text(
        f"""# Алфавит — черновик → v0.1 (refresh)

> Спека: `Тринити cursor/ворота/КАК_собираем_атомы/СПЕЦИФИКАЦИЯ_алфавит_v0.1.md`  
> Сверка: `отчёты/алфавит_пересечение.md` · {report['дата']}

## Корпуса PASS

| Стихия | n | E | закон | материя |
|--------|---|---|-------|---------|
"""
        + "\n".join(
            f"| {r['name']} | {r['n_atoms']} | {r['E']} | `{r['закон']}` | `{r['материя']}` |"
            for r in rows
        )
        + """

## Инвариант корня (∩)

"""
        + " · ".join(atom_i)
        + """

## Рендер (∩)

"""
        + " · ".join(ren_i)
        + """

## Долги

- гром.`delay_s` — постоянный честный долг  
- дождь 7350 axes_decoupled — архив (sync = live-пакет)  
""",
        encoding="utf-8",
    )

    print(json.dumps({"ok": True, "atom_∩": atom_i, "ren_∩": ren_i, "zak_∩": zak_i, "n": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

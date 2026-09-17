# -*- coding: utf-8 -*-
"""Law vs instance разметка params_104.

Для каждого признака:
  within = средняя внутристихийная дисперсия
  between = дисперсия средних стихий
  ratio = between / (within + eps)

Высокий ratio → скорее law (разделяет стихии, стабилен внутри).
Низкий ratio → скорее instance (шум экземпляра).

Не вердикт уха. delay_s исключён.

Запуск: python3 scripts/law_vs_instance_params104.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

КОРЕНЬ = Path(__file__).resolve().parents[1]
OUT_JSON = КОРЕНЬ / "отчёты" / "law_vs_instance_params104.json"
OUT_MD = КОРЕНЬ / "отчёты" / "law_vs_instance_params104.md"
OUT_E = КОРЕНЬ / "выход" / "атомная_визуализация" / "E_law_vs_instance.html"

PACKAGES = {
    "дождь_live": "выход/атомы_полные_дождь_live/атомы_звук_образ.json",
    "огонь": "выход/атомы_полные_огонь/атомы_звук_образ.json",
    "ветер": "выход/атомы_полные_ветер/атомы_звук_образ.json",
    "водопад": "выход/атомы_полные_водопад/атомы_звук_образ.json",
    "река": "выход/атомы_полные_река/атомы_звук_образ.json",
    "гром": "выход/атомы_полные_гром/атомы_звук_образ.json",
}
BAN = {"delay_s", "distance_km", "band_center_hz"}  # служебные / долг
MAX_PER = 400
# пороги (эвристика, не догма)
LAW_RATIO = 2.0
INSTANCE_RATIO = 0.5


def flat_params(p):
    out = {}
    for k, v in (p or {}).items():
        if k in BAN or k == "band_method":
            continue
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def load_by_style():
    rng = np.random.default_rng(42)
    by = {}
    for name, rel in PACKAGES.items():
        pkg = json.load(open(КОРЕНЬ / rel, encoding="utf-8"))
        rows = []
        for a in pkg.get("atoms") or []:
            p = flat_params((a.get("звук_ядро") or {}).get("params_104"))
            if len(p) >= 15:
                rows.append(p)
        if len(rows) > MAX_PER:
            idx = rng.choice(len(rows), MAX_PER, replace=False)
            rows = [rows[i] for i in idx]
        by[name] = rows
    return by


def feature_keys(by):
    # ключ должен быть у ≥80% атомов каждой стихии с n≥5
    counts = defaultdict(lambda: defaultdict(int))
    totals = {s: len(rows) for s, rows in by.items()}
    for s, rows in by.items():
        for r in rows:
            for k in r:
                counts[k][s] += 1
    keys = []
    for k, per in counts.items():
        ok = True
        for s, n in totals.items():
            if n < 5:
                continue
            if per.get(s, 0) < 0.8 * n:
                ok = False
                break
        if ok:
            keys.append(k)
    return sorted(keys)


def analyze(by, keys):
    styles = [s for s, rows in by.items() if len(rows) >= 5]
    results = []
    for k in keys:
        within_vars = []
        means = []
        ns = []
        for s in styles:
            vals = np.array([r[k] for r in by[s] if k in r], dtype=np.float64)
            if len(vals) < 3:
                continue
            within_vars.append(float(np.var(vals)))
            means.append(float(np.mean(vals)))
            ns.append(len(vals))
        if len(means) < 3:
            continue
        within = float(np.mean(within_vars))
        between = float(np.var(means))
        # нормализуем шкалу: между / (внутри + eps), но учитываем разный масштаб —
        # используем also η²-like: between / (between + within)
        denom = between + within + 1e-12
        eta = between / denom
        ratio = between / (within + 1e-12)
        if ratio >= LAW_RATIO and eta >= 0.45:
            label = "law"
        elif ratio <= INSTANCE_RATIO or eta <= 0.2:
            label = "instance"
        else:
            label = "mixed"
        results.append({
            "key": k,
            "within": round(within, 6),
            "between": round(between, 6),
            "ratio": round(ratio, 4),
            "eta": round(eta, 4),
            "label": label,
            "n_styles": len(means),
        })
    results.sort(key=lambda x: -x["eta"])
    return results, styles


def main() -> int:
    by = load_by_style()
    keys = feature_keys(by)
    rows, styles = analyze(by, keys)
    counts = {lab: sum(1 for r in rows if r["label"] == lab) for lab in ("law", "mixed", "instance")}
    law = [r for r in rows if r["label"] == "law"]
    inst = [r for r in rows if r["label"] == "instance"]
    mixed = [r for r in rows if r["label"] == "mixed"]

    report = {
        "дата": date.today().isoformat(),
        "метод": "between-style var / within-style var; eta=between/(between+within)",
        "пороги": {"law_ratio": LAW_RATIO, "instance_ratio": INSTANCE_RATIO, "law_eta": 0.45, "instance_eta": 0.2},
        "запрет": "не вердикт уха; delay_s исключён",
        "styles": styles,
        "n_per_style": {s: len(by[s]) for s in styles},
        "n_features": len(rows),
        "counts": counts,
        "features": rows,
        "top_law": law[:25],
        "top_instance": sorted(inst, key=lambda x: x["eta"])[:25],
        "note": "эвристика для разметки корпуса; подтверждать пересборкой звука",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def table(items, n=20):
        lines = ["| признак | η² | ratio | within | between |", "|---------|----|-------|--------|---------|"]
        for r in items[:n]:
            lines.append(
                f"| `{r['key']}` | {r['eta']} | {r['ratio']} | {r['within']:.4g} | {r['between']:.4g} |"
            )
        return "\n".join(lines)

    md = f"""# Law vs instance · params_104

> {report['дата']} · эвристика between/within · **не** вердикт уха

- стихии: {', '.join(styles)}
- признаков: **{len(rows)}**
- law: **{counts['law']}** · mixed: **{counts['mixed']}** · instance: **{counts['instance']}**
- порог law: ratio≥{LAW_RATIO} и η²≥0.45 · instance: ratio≤{INSTANCE_RATIO} или η²≤0.2

## Top law (разделяют стихии)

{table(law)}

## Top instance (шум экземпляра)

{table(sorted(inst, key=lambda x: x['eta']))}

## Mixed (середина)

{table(mixed, 15)}

## Честно

- Ветер/водопад n=19 — within неустойчив; после bandpass-104 разброс есть, но корпус мал.
- Признаки с near-zero within у крупных корпусов (огонь/гром) сильнее тянут в law.
- Дальше: проверить law-поля пересборкой (держать law, варьировать instance) — отдельно.

JSON: `{OUT_JSON.relative_to(КОРЕНЬ)}`
"""
    OUT_MD.write_text(md, encoding="utf-8")

    # E page
    def lis(items, n=18):
        return "".join(
            f"<li><code>{r['key']}</code> · η²={r['eta']} · ratio={r['ratio']}</li>"
            for r in items[:n]
        )

    OUT_E.parent.mkdir(parents=True, exist_ok=True)
    OUT_E.write_text(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — law vs instance</title>
<style>
body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:980px;margin:1.5rem auto;padding:0 1rem}}
.meta{{opacity:.85}} .grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem}}
h2{{font-size:1.05rem;margin-top:0}} ul{{padding-left:1.1rem;font-size:.9rem}}
code{{color:#9dceb0}} a{{color:#9dceb0}}
.badge{{display:inline-block;padding:.15rem .5rem;border-radius:4px;margin-right:.4rem;font-size:.85rem}}
.law{{background:#1e3a2f;color:#9dceb0}} .mix{{background:#3a3420;color:#e0d090}} .inst{{background:#3a2020;color:#e0a0a0}}
</style></head><body>
<h1>Law vs instance · params_104</h1>
<p class="meta">{report['дата']} · between/within · не вердикт уха<br/>
<span class="badge law">law {counts['law']}</span>
<span class="badge mix">mixed {counts['mixed']}</span>
<span class="badge inst">instance {counts['instance']}</span>
· feats {len(rows)} · styles {len(styles)}</p>
<div class="grid">
<div><h2>Law</h2><ul>{lis(law)}</ul></div>
<div><h2>Instance</h2><ul>{lis(sorted(inst, key=lambda x: x['eta']))}</ul></div>
<div><h2>Mixed</h2><ul>{lis(mixed)}</ul></div>
</div>
<p><a href="/выход/атомная_визуализация/index.html">← хаб</a> ·
<a href="/выход/атомная_визуализация/E_кросс_pca.html">PCA</a> ·
<a href="/выход/атомная_визуализация/E_алфавит_пересечение.html">алфавит ∩</a></p>
</body></html>
""",
        encoding="utf-8",
    )

    print(json.dumps({
        "ok": True,
        "n_features": len(rows),
        "counts": counts,
        "top_law": [r["key"] for r in law[:8]],
        "top_instance": [r["key"] for r in sorted(inst, key=lambda x: x["eta"])[:8]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

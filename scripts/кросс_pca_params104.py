# -*- coding: utf-8 -*-
"""Кросс-стихийный PCA по params_104 PASS-пакетов.

Без sklearn/umap: PCA через SVD (numpy).
Правило: не сливать число и ухо; PCA = карта, не вердикт.

Запуск: python3 scripts/кросс_pca_params104.py
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

КОРЕНЬ = Path(__file__).resolve().parents[1]
OUT_JSON = КОРЕНЬ / "отчёты" / "кросс_pca_params104.json"
OUT_MD = КОРЕНЬ / "отчёты" / "кросс_pca_params104.md"
OUT_E = КОРЕНЬ / "выход" / "атомная_визуализация" / "E_кросс_pca.html"
OUT_PNG = КОРЕНЬ / "выход" / "атомная_визуализация" / "кросс_pca.png"

# только корпуса, валидные для кросс (PASS / рабочий; гром без delay в фичах)
PACKAGES = {
    "дождь_live": "выход/атомы_полные_дождь_live/атомы_звук_образ.json",
    "огонь": "выход/атомы_полные_огонь/атомы_звук_образ.json",
    "ветер": "выход/атомы_полные_ветер/атомы_звук_образ.json",
    "водопад": "выход/атомы_полные_водопад/атомы_звук_образ.json",
    "река": "выход/атомы_полные_река/атомы_звук_образ.json",
    "гром": "выход/атомы_полные_гром/атомы_звук_образ.json",
}

# не брать в PCA дырявые/несопоставимые ключи
BAN_KEYS = {
    "delay_s", "distance_km",  # гром долг
}

COLORS = {
    "дождь_live": "#6a9fc9",
    "огонь": "#e07840",
    "ветер": "#8aa4b8",
    "водопад": "#7a9eb0",
    "река": "#6a90a8",
    "гром": "#c4b06a",
}

MAX_PER = 400  # баланс: не дать огню/грому задавить


def flatten_params(p: dict) -> dict[str, float]:
    out = {}
    for k, v in (p or {}).items():
        if k in BAN_KEYS:
            continue
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def load_rows():
    rows = []
    rng = np.random.default_rng(42)
    for name, rel in PACKAGES.items():
        path = КОРЕНЬ / rel
        pkg = json.load(open(path, encoding="utf-8"))
        atoms = pkg.get("atoms") or []
        cand = []
        for a in atoms:
            p104 = (a.get("звук_ядро") or {}).get("params_104")
            if not p104:
                continue
            flat = flatten_params(p104)
            if len(flat) < 20:
                continue
            cand.append(flat)
        if not cand:
            continue
        if len(cand) > MAX_PER:
            idx = rng.choice(len(cand), size=MAX_PER, replace=False)
            cand = [cand[i] for i in idx]
        for flat in cand:
            rows.append({"стихия": name, "params": flat})
    return rows


def build_matrix(rows):
    # ключи = пересечение, покрытое ≥80% строк
    key_count = defaultdict(int)
    for r in rows:
        for k in r["params"]:
            key_count[k] += 1
    thr = int(0.8 * len(rows))
    keys = sorted(k for k, n in key_count.items() if n >= thr)
    X = np.zeros((len(rows), len(keys)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            X[i, j] = r["params"].get(k, np.nan)
    # импутация медианой колонки
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[np.isnan(col)] = med
        X[:, j] = col
    return X, keys


def pca_svd(Z, n_comp=3):
    # Z: n×d standardized, finite
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = (S ** 2) / max(1, Z.shape[0] - 1)
    total = float(ev.sum()) + 1e-12
    ratio = ev / total
    comps = Vt[:n_comp]
    scores = Z @ comps.T
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores, ratio, comps


def standardize(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    # выкинуть почти-константные колонки
    keep = sd > 1e-8
    if keep.sum() < 5:
        keep = np.ones(X.shape[1], dtype=bool)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z = (X - mu) / sd
        return Z, mu, sd, keep
    Xk = X[:, keep]
    muk = mu[keep]
    sdk = sd[keep]
    Z = (Xk - muk) / sdk
    Z = np.clip(Z, -20, 20)
    return Z, muk, sdk, keep


def loadings_top(comps, keys, ratio, k=8):
    out = []
    for i, comp in enumerate(comps):
        order = np.argsort(np.abs(comp))[::-1][:k]
        out.append({
            "pc": i + 1,
            "var": round(float(ratio[i]), 4),
            "top": [
                {"key": keys[j], "w": round(float(comp[j]), 4)}
                for j in order
            ],
        })
    return out


def centroids(scores, labels):
    out = {}
    for name in sorted(set(labels)):
        m = scores[np.array(labels) == name]
        out[name] = {
            "n": int(m.shape[0]),
            "pc1": round(float(m[:, 0].mean()), 4),
            "pc2": round(float(m[:, 1].mean()), 4),
            "pc3": round(float(m[:, 2].mean()), 4) if m.shape[1] > 2 else None,
            "std1": round(float(m[:, 0].std()), 4),
            "std2": round(float(m[:, 1].std()), 4),
        }
    return out


def pairwise_centroid_dist(cent):
    names = list(cent.keys())
    dist = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            da = np.array([cent[a]["pc1"], cent[a]["pc2"]])
            db = np.array([cent[b]["pc1"], cent[b]["pc2"]])
            dist.append({
                "a": a, "b": b,
                "d_pc12": round(float(np.linalg.norm(da - db)), 4),
            })
    dist.sort(key=lambda x: x["d_pc12"])
    return dist


def save_scatter_png(scores, labels, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return f"no matplotlib: {e}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 7), dpi=140)
    ax.set_facecolor("#0e1218")
    fig.patch.set_facecolor("#0e1218")
    for name in sorted(set(labels)):
        m = np.array(labels) == name
        ax.scatter(
            scores[m, 0], scores[m, 1],
            s=14, alpha=0.55, c=COLORS.get(name, "#ccc"),
            label=f"{name} (n={m.sum()})", edgecolors="none",
        )
    ax.set_xlabel("PC1", color="#c8d2dc")
    ax.set_ylabel("PC2", color="#c8d2dc")
    ax.tick_params(colors="#8a96a3")
    for spine in ax.spines.values():
        spine.set_color("#2a3340")
    leg = ax.legend(frameon=False, fontsize=8, loc="best")
    for t in leg.get_texts():
        t.set_color("#c8d2dc")
    ax.set_title("params_104 · PCA (PASS корпуса)", color="#e8eef6", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def main() -> int:
    rows = load_rows()
    if len(rows) < 30:
        raise SystemExit(f"мало строк: {len(rows)}")
    X, keys = build_matrix(rows)
    Z, _, _, keep = standardize(X)
    keys = [k for k, m in zip(keys, keep) if m]
    scores, ratio, comps = pca_svd(Z, n_comp=3)
    labels = [r["стихия"] for r in rows]
    loads = loadings_top(comps, keys, ratio, k=10)
    cent = centroids(scores, labels)
    dists = pairwise_centroid_dist(cent)

    # subsample points for E page (не всё)
    rng = np.random.default_rng(7)
    pts = []
    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[lab].append(i)
    for lab, idxs in by.items():
        take = idxs if len(idxs) <= 80 else list(rng.choice(idxs, 80, replace=False))
        for i in take:
            pts.append({
                "s": lab,
                "x": round(float(scores[i, 0]), 4),
                "y": round(float(scores[i, 1]), 4),
            })

    report = {
        "дата": date.today().isoformat(),
        "метод": "PCA SVD на standardized params_104; без umap/sklearn",
        "запрет": "не вердикт уха/глаза; delay_s исключён",
        "n_rows": len(rows),
        "n_features": len(keys),
        "max_per_style": MAX_PER,
        "var_explained": [round(float(x), 4) for x in ratio[:8]],
        "var_pc1_pc2": round(float(ratio[0] + ratio[1]), 4),
        "loadings": loads,
        "centroids": cent,
        "distances_pc12_asc": dists,
        "counts": {k: int(v["n"]) for k, v in cent.items()},
        "features_used_n": len(keys),
        "features_sample": keys[:40],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    png_note = save_scatter_png(scores, labels, OUT_PNG)

    # markdown
    lines = [
        "# Кросс PCA · params_104",
        "",
        f"> {report['дата']} · карта, не вердикт уха/глаза · delay_s исключён",
        "",
        f"- строк: **{report['n_rows']}** (≤{MAX_PER}/стихия)",
        f"- признаков (≥80% покрытие): **{report['n_features']}**",
        f"- PC1+PC2: **{report['var_pc1_pc2']}** дисперсии",
        f"- PNG: `{OUT_PNG.relative_to(КОРЕНЬ) if OUT_PNG.exists() else png_note}`",
        "",
        "## Центроиды (PC1, PC2)",
        "",
        "| Стихия | n | PC1 | PC2 | σ1 | σ2 |",
        "|--------|---|-----|-----|----|----|",
    ]
    for name, c in cent.items():
        lines.append(
            f"| {name} | {c['n']} | {c['pc1']} | {c['pc2']} | {c['std1']} | {c['std2']} |"
        )
    lines += [
        "",
        "## Ближайшие пары (d в PC1–PC2)",
        "",
    ]
    for d in dists[:8]:
        lines.append(f"- {d['a']} ↔ {d['b']}: **{d['d_pc12']}**")
    lines += ["", "## Loadings PC1 (top)", ""]
    for item in loads[0]["top"][:8]:
        lines.append(f"- `{item['key']}`: {item['w']}")
    lines += ["", "## Loadings PC2 (top)", ""]
    for item in loads[1]["top"][:8]:
        lines.append(f"- `{item['key']}`: {item['w']}")
    lines += [
        "",
        "## Честно",
        "",
        "- PCA показывает **разделимость спектральных/104-осей**, не узнаваемость образа.",
        "- Мало атомов у ветра/водопада → центроид менее устойчив.",
        "- Гром в пространстве без `delay_s` (долг).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # E HTML — интерактивный scatter
    pts_json = json.dumps(pts, ensure_ascii=False)
    colors_json = json.dumps(COLORS, ensure_ascii=False)
    cent_rows = "".join(
        f"<tr><td>{n}</td><td>{c['n']}</td><td>{c['pc1']}</td><td>{c['pc2']}</td>"
        f"<td>{c['std1']}</td><td>{c['std2']}</td></tr>"
        for n, c in cent.items()
    )
    dist_li = "".join(f"<li>{d['a']} ↔ {d['b']}: <b>{d['d_pc12']}</b></li>" for d in dists[:6])
    load1 = "".join(f"<li><code>{x['key']}</code> {x['w']}</li>" for x in loads[0]["top"][:6])
    load2 = "".join(f"<li><code>{x['key']}</code> {x['w']}</li>" for x in loads[1]["top"][:6])

    OUT_E.parent.mkdir(parents=True, exist_ok=True)
    OUT_E.write_text(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — кросс PCA params_104</title>
<style>
body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:1100px;margin:1.5rem auto;padding:0 1rem}}
.meta{{opacity:.85;margin-bottom:1rem}}
canvas{{width:100%;max-width:920px;height:560px;background:#121820;border:1px solid #2a3340;border-radius:8px}}
table{{width:100%;border-collapse:collapse;font-size:.9rem;margin:1rem 0}}
th,td{{border-bottom:1px solid #2a3340;padding:.4rem .3rem;text-align:left}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}}
code{{color:#9dceb0}} a{{color:#9dceb0}}
.legend span{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}}
</style></head><body>
<h1>Кросс PCA · params_104</h1>
<p class="meta">{report['дата']} · n={report['n_rows']} · feats={report['n_features']} · PC1+PC2={report['var_pc1_pc2']}<br/>
карта чисел, <b>не</b> вердикт уха/глаза · <code>delay_s</code> исключён</p>
<canvas id="c" width="920" height="560"></canvas>
<p class="legend" id="leg"></p>
<div class="grid">
<div>
<h2>Центроиды</h2>
<table><thead><tr><th>Стихия</th><th>n</th><th>PC1</th><th>PC2</th><th>σ1</th><th>σ2</th></tr></thead>
<tbody>{cent_rows}</tbody></table>
</div>
<div>
<h2>Ближайшие пары</h2>
<ul>{dist_li}</ul>
<h2>PC1 loadings</h2><ul>{load1}</ul>
<h2>PC2 loadings</h2><ul>{load2}</ul>
</div>
</div>
<p><a href="/выход/атомная_визуализация/index.html">← хаб</a> ·
<a href="/выход/атомная_визуализация/E_алфавит_пересечение.html">алфавит ∩</a></p>
<script>
const pts = {pts_json};
const colors = {colors_json};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const xs = pts.map(p=>p.x), ys = pts.map(p=>p.y);
const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
const pad=36;
function sx(x){{return pad + (x-xmin)/(xmax-xmin+1e-9)*(canvas.width-2*pad);}}
function sy(y){{return canvas.height-pad - (y-ymin)/(ymax-ymin+1e-9)*(canvas.height-2*pad);}}
ctx.clearRect(0,0,canvas.width,canvas.height);
ctx.strokeStyle='#2a3340'; ctx.beginPath();
ctx.moveTo(pad, canvas.height-pad); ctx.lineTo(canvas.width-pad, canvas.height-pad);
ctx.moveTo(pad, pad); ctx.lineTo(pad, canvas.height-pad); ctx.stroke();
for (const p of pts) {{
  ctx.fillStyle = colors[p.s] || '#ccc';
  ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), 2.6, 0, Math.PI*2); ctx.fill();
}}
const leg = document.getElementById('leg');
leg.innerHTML = Object.keys(colors).map(k=>`<span style="background:${{colors[k]}}"></span>${{k}}`).join(' · ');
</script>
</body></html>
""",
        encoding="utf-8",
    )

    print(json.dumps({
        "ok": True,
        "n_rows": report["n_rows"],
        "n_features": report["n_features"],
        "var_pc12": report["var_pc1_pc2"],
        "closest": dists[:3],
        "png": png_note,
        "E": str(OUT_E.relative_to(КОРЕНЬ)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

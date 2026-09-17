# -*- coding: utf-8 -*-
"""CP пути 2 — приёмка кирпич-момента (НЕ d₉ / НЕ CP#2).

Проверки:
  1) Прямой round-trip: wav → в_кирпичи → из_кирпичей → corr с оригиналом
  2) Библиотека: npz+старт → из_кирпичей (детерминизм sha)
  3) Организм ШТОРМ: детерминизм sha
  4) Unit: синус 440 Гц → corr ≈ 1

Пороги (честные, без нормировки метрики):
  - буквы / тон: corr ≥ 0.995
  - природа / шум: corr ≥ 0.990
  - корпус: доля PASS ≥ 0.95 (≈52/55)

Запуск: python3 scripts/cp_моментов.py
Отчёт:  отчёты/cp_моментов.md
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import добыча_словаря as Д  # noqa
import ядро.словарь_кирпичей as СЛ  # noqa
from ядро.кирпич_момент import SR, ОКНО, в_кирпичи, из_кирпичей  # noqa

БИБ = os.path.join(КОРЕНЬ, "данные", "библиотека_кирпичей")
ОТЧЁТ = os.path.join(КОРЕНЬ, "отчёты", "cp_моментов.md")

ПОРОГ_ТОН = 0.995
ПОРОГ_ШУМ = 0.990
ДОЛЯ_PASS = 0.95

# природа / шум в корпусе — чуть мягче порог (overlap-add на шумах)
_ШУМ_ID = {
    "etalon_dozhd", "etalon_grom", "etalon_veter", "etalon_pesok",
    "etalon_ogon", "etalon_vodopad", "etalon_kashel", "etalon_chihanie",
    "etalon_hrap", "etalon_dyhanie", "etalon_serdce", "etalon_komar",
    "etalon_lyagushki",
}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.dot(a, a))
    db = float(np.dot(b, b))
    if da < 1e-18 or db < 1e-18:
        return 0.0
    return float(np.dot(a, b) / np.sqrt(da * db))


def _порог(cid: str) -> float:
    if cid in _ШУМ_ID:
        return ПОРОГ_ШУМ
    return ПОРОГ_ТОН


def _load_cid(cid: str) -> tuple[np.ndarray, int] | None:
    c = Д._cell(cid)
    ap = Д._audio_path(c)
    if not ap:
        return None
    x, sr = Д._load_audio(ap)
    x = np.asarray(x, dtype=np.float64)
    if sr != SR:
        x = np.interp(
            np.linspace(0, len(x) - 1, int(len(x) * SR / sr)),
            np.arange(len(x)), x,
        )
        sr = SR
    peak = np.max(np.abs(x)) + 1e-12
    return x / peak, sr


def roundtrip_прямой(cid: str) -> dict:
    loaded = _load_cid(cid)
    if not loaded:
        return {"id": cid, "ok": False, "corr": 0.0, "err": "нет audio"}
    x, sr = loaded
    bricks = в_кирпичи(x, sr)
    y = из_кирпичей(bricks, длина=len(x), sr=sr)
    c = _corr(x, y)
    thr = _порог(cid)
    return {
        "id": cid, "ok": c >= thr, "corr": c, "порог": thr,
        "атомов": len(bricks), "сэмплов": len(x),
    }


def roundtrip_синус() -> dict:
    t = np.arange(int(0.5 * SR)) / SR
    x = np.sin(2 * np.pi * 440.0 * t)
    bricks = в_кирпичи(x, SR)
    y = из_кирпичей(bricks, длина=len(x), sr=SR)
    c = _corr(x, y)
    return {"id": "sine_440", "ok": c >= 0.999, "corr": c, "атомов": len(bricks)}


def библиотека_детерминизм(cid: str = "etalon_dozhd") -> dict:
    import json
    man_path = os.path.join(БИБ, "манифест.json")
    npz_path = os.path.join(БИБ, cid + ".npz")
    if not (os.path.isfile(man_path) and os.path.isfile(npz_path)):
        return {"id": cid, "ok": False, "err": "библиотека не найдена"}
    M = json.load(open(man_path, encoding="utf-8"))
    org = next((o for o in M["органы"] if o["источник"] == cid), None)
    if not org:
        return {"id": cid, "ok": False, "err": "нет органа"}
    z = np.load(npz_path)
    bricks = [
        {"старт": a["старт"], "спектр_магнитуда": z["mag"][j], "спектр_фаза": z["phase"][j]}
        for j, a in enumerate(org["атомы"])
    ]
    nmax = max(b["старт"] for b in bricks) + ОКНО + 10
    y1 = из_кирпичей(bricks, длина=nmax, sr=SR)
    y2 = из_кирпичей(bricks, длина=nmax, sr=SR)
    h1 = hashlib.sha256(y1.tobytes()).hexdigest()[:16]
    h2 = hashlib.sha256(y2.tobytes()).hexdigest()[:16]
    return {"id": cid, "ok": h1 == h2, "sha": h1, "атомов": len(bricks)}


def шторм_детерминизм() -> dict:
    from ядро.организм import загрузить_манифест, собрать_по_имени
    man = загрузить_манифест()
    s1 = собрать_по_имени("шторм", органы=man)
    s2 = собрать_по_имени("шторм", органы=man)
    h1 = hashlib.sha256(s1.tobytes()).hexdigest()[:16]
    h2 = hashlib.sha256(s2.tobytes()).hexdigest()[:16]
    return {
        "ok": h1 == h2, "sha": h1,
        "длит_с": round(len(s1) / SR, 3),
    }


def main() -> None:
    t0 = time.time()
    rows = []
    ids = СЛ.id_корпуса()
    for cid in ids:
        try:
            rows.append(roundtrip_прямой(cid))
        except Exception as e:  # noqa
            rows.append({"id": cid, "ok": False, "corr": 0.0, "err": str(e)})

    sine = roundtrip_синус()
    lib = библиотека_детерминизм()
    try:
        storm = шторм_детерминизм()
    except Exception as e:  # noqa
        storm = {"ok": False, "err": str(e)}

    n_ok = sum(1 for r in rows if r.get("ok"))
    n = len(rows)
    доля = n_ok / n if n else 0.0
    med = float(np.median([r["corr"] for r in rows])) if rows else 0.0
    корпус_ok = доля >= ДОЛЯ_PASS
    all_ok = корпус_ok and sine["ok"] and lib.get("ok") and storm.get("ok")

    lines = [
        "# CP моментов (путь 2)",
        "",
        f"> Дата: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"> Код: `ядро/кирпич_момент.py` · метрика: **corr**, не d₉",
        "",
        f"## Вердикт: {'**PASS**' if all_ok else '**FAIL**'}",
        "",
        "| Проверка | Результат |",
        "|----------|-----------|",
        f"| Синус 440 Гц corr≥0.999 | {'PASS' if sine['ok'] else 'FAIL'} ({sine['corr']:.6f}) |",
        f"| Корпус round-trip ≥{ДОЛЯ_PASS:.0%} | "
        f"{'PASS' if корпус_ok else 'FAIL'} ({n_ok}/{n} = {доля:.1%}, медиана corr={med:.4f}) |",
        f"| Библиотека детерминизм ({lib.get('id','?')}) | "
        f"{'PASS' if lib.get('ok') else 'FAIL'} sha={lib.get('sha', lib.get('err'))} |",
        f"| ШТОРМ детерминизм | "
        f"{'PASS' if storm.get('ok') else 'FAIL'} sha={storm.get('sha', storm.get('err'))} |",
        "",
        "### Пороги",
        f"- тон / буквица: corr ≥ {ПОРОГ_ТОН}",
        f"- природа / шум: corr ≥ {ПОРОГ_ШУМ}",
        "",
        "## Корпус (по клетке)",
        "",
        "| id | corr | порог | ok | атомов |",
        "|----|------|-------|----|--------|",
    ]
    for r in sorted(rows, key=lambda x: x.get("corr", 0)):
        err = r.get("err", "")
        lines.append(
            f"| {r['id']} | {r.get('corr', 0):.4f} | {r.get('порог', 0):.3f} | "
            f"{'✓' if r.get('ok') else '✗'} | {r.get('атомов', '—')} |"
            + (f" <!-- {err} -->" if err else "")
        )
    lines += [
        "",
        f"Время прогона: {time.time() - t0:.1f}с",
        "",
        "## Заметки",
        "- Это **не** CP#2 (`словарь_кирпичей`). См. `ДВЕ_СИСТЕМЫ_КИРПИЧЕЙ.md`.",
        "- Библиотека фильтрует тишину (rms) — прямой round-trip здесь идёт по полным `в_кирпичи`, не по npz.",
    ]
    os.makedirs(os.path.dirname(ОТЧЁТ), exist_ok=True)
    open(ОТЧЁТ, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{'PASS' if all_ok else 'FAIL'} · корпус {n_ok}/{n} · медиана {med:.4f}")
    print(f"отчёт → {ОТЧЁТ}")
    if not sine["ok"]:
        print("  sine FAIL", sine)
    if not lib.get("ok"):
        print("  lib FAIL", lib)
    if not storm.get("ok"):
        print("  storm FAIL", storm)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

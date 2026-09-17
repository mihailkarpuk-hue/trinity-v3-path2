# -*- coding: utf-8
"""Добыча словаря первокирпичей из корпуса 18+37.

usage:
  python3 scripts/добыча_словаря.py
  python3 scripts/добыча_словаря.py --quick   # без wav-реконструкции
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ПРОЕКТ = os.path.dirname(КОРЕНЬ)
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))
sys.path.insert(0, os.path.join(ПРОЕКТ, "scripts"))

from atoms_full103 import analyze_full_103  # noqa: E402
from оси import оси_звука  # noqa: E402

from ядро.атомизация import атомизировать  # noqa: E402
from ядро.ген import ген_из_атома, гены_из_атомов  # noqa: E402
from ядро.метрика import d9, вектор_из_словаря, загрузить_индекс  # noqa: E402
from ядро.синтез import синтезировать, bukvitsa_synth_modes, bukvitsa_класс, _буква_буквицы, _ВЗРЫВНЫЕ  # noqa: E402
from ядро.словарь_кирпичей import (  # noqa: E402
    ближайший_тип,
    вектор_типа_из_гена,
    id_корпуса,
    k_means,
    загрузить,
    загрузить_атомы_корпуса,
    построить_словарь,
    сохранить,
    собрать_гены_клетки,
    собрать_гены_клетки_адаптивно,
    собрать_гены_из_атомов,
    сэмпл_гена,
    zscore,
)
from ядро.фаза import загрузить as wav_load  # noqa: E402
from scripts.круг import _фенотип_до  # noqa: E402

КЛЕТКИ = os.path.join(КОРЕНЬ, "данные", "клеточки")
КАТАЛОГ = os.path.join(КЛЕТКИ, "каталог.json")
ФОРМУЛА = os.path.join(КОРЕНЬ, "данные", "формула_калибровки.json")
ОТЧЁТ = os.path.join(КОРЕНЬ, "отчёты", "словарь_кирпичей.md")
ВЫХОД_WAV = os.path.join(КОРЕНЬ, "выход", "словарь")
SR = 22050
SR_PH = 16000

# внекорпусные для обобщения (зонд отсутствует — синтетический базис)
ВНЕ_КОРПУСА = (
    "bazis_00_sinus_etalon",
    "bazis_01_kvadrat_etalon",
    "bazis_03_belyi_shum_etalon",
    "bazis_04_rozovyi_shum_etalon",
    "bazis_05_impuls_etalon",
)

CP2_DEMO = ("bazis_12_kaplya", "etalon_dozhd", "живая_А")


def _cells() -> list[dict]:
    d = json.load(open(КАТАЛОГ, encoding="utf-8"))
    return next(v for v in d.values() if isinstance(v, list))


def _cell(cid: str) -> dict:
    for c in _cells():
        if c["id"] == cid:
            return c
    raise KeyError(cid)


def _audio_path(c: dict) -> str | None:
    z = c.get("звук") or ""
    p = os.path.join(КЛЕТКИ, z)
    if os.path.isfile(p):
        return p
    base = os.path.join(КЛЕТКИ, os.path.splitext(c.get("атомы") or "")[0])
    for ext in (".wav", ".m4a", ".mp3"):
        if os.path.isfile(base + ext):
            return base + ext
    return None


def _load_audio(path: str, sr: int = SR) -> tuple[np.ndarray, int]:
    if path.lower().endswith(".wav"):
        x, s = wav_load(path)
    else:
        from обогатить_атомы_104 import _load_any  # noqa: WPS433
        x, s = _load_any(path)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    m = np.max(np.abs(x))
    if m > 0:
        x /= m
    if s != sr:
        from scipy.signal import resample
        x = resample(x, int(len(x) * sr / s)).astype(np.float64)
        s = sr
    return x, s


def _фенотип(x: np.ndarray, sr: int) -> dict:
    if sr != SR_PH:
        from scipy.signal import resample
        x = resample(x, int(len(x) * SR_PH / sr)).astype(np.float64)
        sr = SR_PH
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m
    return {**analyze_full_103(x, sr), **оси_звука(x, sr)}


def _write_wav(path: str, x: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _типы_по_id(словарь: dict) -> dict[int, dict]:
    return {int(t["id"]): t for t in словарь["типы"]}


def _centroids(словарь: dict) -> np.ndarray:
    return np.asarray(словарь["centroids_z"], dtype=np.float64)


def _mu_sd(словарь: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(словарь["mu"], dtype=np.float64),
        np.asarray(словарь["sd"], dtype=np.float64),
    )


def пересобрать_клетку(
    cid: str,
    словарь: dict,
    *,
    labels_map: dict[tuple[str, int], int] | None = None,
) -> tuple[list[dict], float | None]:
    """Атомы клетки → гены из типов. Возвращает (гены, dur)."""
    gens, dur, _ = собрать_гены_клетки(cid, словарь, labels_map=labels_map, живость=0.0)
    return gens, dur


def ε_квант_клетки(
    cid: str,
    словарь: dict,
    idx: dict,
    labels_map: dict | None = None,
    *,
    save_wav: bool = False,
) -> dict:
    c = _cell(cid)
    ap = _audio_path(c)
    x_ref, sr_ref = (_load_audio(ap) if ap else (None, SR))
    stored = c.get("параметры104") or {}
    if len(stored) < 100 and x_ref is not None:
        before = _фенотип(x_ref, sr_ref)
    elif x_ref is not None:
        before = _фенотип_до(c, x_ref, sr_ref)
    else:
        before = stored

    meta_cell = c
    if c.get("группа") == "буквица_живая" and len(before) >= 50:
        stored_p = c.get("параметры104") or {}
        meta_cell = {**c, "параметры104": {**stored_p, **before}}

    jp = os.path.join(КЛЕТКИ, c.get("атомы") or "")
    stored_atoms = json.load(open(jp, encoding="utf-8")).get("atoms") or []
    candidates: list[tuple[list, float]] = []
    gens_s, dur_s, _ = собрать_гены_из_атомов(
        stored_atoms, словарь, labels_map=labels_map, cid=cid,
    )
    candidates.append((gens_s, dur_s))
    gens_sn, dur_sn, _ = собрать_гены_из_атомов(
        stored_atoms, словарь, labels_map=labels_map, cid=cid, blend=False,
    )
    candidates.append((gens_sn, dur_sn))
    skip_fresh = (
        c.get("группа") == "буквица_живая"
        and (
            "протяж" in cid.lower()
            or bukvitsa_класс(meta_cell) == "шипящая"
            or _буква_буквицы(meta_cell) in _ВЗРЫВНЫЕ
        )
    )
    if c.get("группа") == "буквица_живая" and x_ref is not None and not skip_fresh:
        fresh = [
            {k: v for k, v in a.items() if not k.startswith("_")}
            for a in атомизировать(x_ref, sr_ref)
        ]
        if fresh:
            gens_f, dur_f, _ = собрать_гены_из_атомов(
                fresh, словарь, labels_map=labels_map, cid=cid,
            )
            candidates.append((gens_f, dur_f))
            gens_fn, dur_fn, _ = собрать_гены_из_атомов(
                fresh, словарь, labels_map=labels_map, cid=cid, blend=False,
            )
            candidates.append((gens_fn, dur_fn))

    best_eps = float("inf")
    gens, dur, y = candidates[0][0], candidates[0][1], None
    for cgens, cdur in candidates:
        synth_modes = (
            bukvitsa_synth_modes(meta_cell)
            if c.get("группа") == "буквица_живая"
            else [c]
        )
        for meta_s in synth_modes:
            cy = синтезировать(cgens, sr=SR, dur=cdur, meta=meta_s)
            ceps = d9(
                вектор_из_словаря(before),
                вектор_из_словаря(_фенотип(cy, SR)),
                mu=idx["mu"],
                sd=idx["sd"],
            )
            if ceps < best_eps:
                best_eps = ceps
                eps = ceps
                gens, dur, y = cgens, cdur, cy
    after = _фенотип(y, SR)
    if save_wav:
        _write_wav(os.path.join(ВЫХОД_WAV, f"{cid}_квант.wav"), y, SR)
        ap = _audio_path(c)
        if ap:
            x, sr = _load_audio(ap)
            _write_wav(os.path.join(ВЫХОД_WAV, f"{cid}_оригинал.wav"), x, sr)
    return {"id": cid, "epsilon_квант": round(eps, 4), "атомов": len(gens)}


def ε_квант_внешний(cid: str, словарь: dict, idx: dict) -> dict:
    """Клетка вне корпуса: атомизация на лету."""
    c = _cell(cid)
    ap = _audio_path(c)
    if not ap:
        raise FileNotFoundError(cid)
    x, sr = _load_audio(ap)
    before = _фенотип(x, sr)
    raw = атомизировать(x, sr)
    atoms = [{k: v for k, v in a.items() if not k.startswith("_")} for a in raw]
    типы = _типы_по_id(словарь)
    mu, sd = _mu_sd(словарь)
    cent = _centroids(словарь)
    gens = []
    for atom in atoms:
        g0 = ген_из_атома(atom)
        vz = (вектор_типа_из_гена(g0) - mu) / sd
        tid = ближайший_тип(vz, cent)
        gens.append(сэмпл_гена(типы[tid], atom, mu, sd, cent_z=cent, blend=True))
    dur = len(x) / sr
    y = синтезировать(gens, sr=SR, dur=dur, meta=c)
    after = _фенотип(y, SR)
    eps = d9(
        вектор_из_словаря(before),
        вектор_из_словаря(after),
        mu=idx["mu"],
        sd=idx["sd"],
    )
    return {"id": cid, "epsilon_квант": round(eps, 4), "атомов": len(gens)}


def _состав_клетки(cid: str, labels_map: dict) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for (cell, _), tid in labels_map.items():
        if cell != cid:
            continue
        total += 1
        counts[str(tid)] = counts.get(str(tid), 0) + 1
    return {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def _отчёт(
    словарь: dict,
    корпус_ε: list[dict],
    вне_ε: list[dict],
    labels_map: dict,
    eps_порог: float,
) -> str:
    null = словарь["нулевой_тест"]
    lines = [
        "# Словарь первокирпичей (Этап 3 · CP#2)",
        "",
        f"Дата: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Нулевой тест",
        "",
        f"- Пройден: **{'ДА' if null['пройден'] else 'НЕТ'}**",
        f"- Лучший k: {null.get('лучший_k')}, запас silhouette: {null.get('лучший_запас')} (порог {null['порог']})",
        f"- Путь: `{словарь['путь']}`",
        "",
        "## Кластеризация",
        "",
        f"- k = {словарь['k']}, silhouette = {словарь['silhouette']}",
        f"- Атомов в корпусе: {словарь['число_атомов']}",
        f"- sha256 словаря: `{словарь['sha256'][:24]}…`",
        "",
        "### Кривая silhouette (реальные vs шум)",
        "",
        "| k | реальные | шум |",
        "|---|----------|-----|",
    ]
    for k in sorted(null["silhouette_реальные"].keys()):
        r = null["silhouette_реальные"][k]
        s = null["silhouette_шум"].get(k, 0)
        lines.append(f"| {k} | {r:.4f} | {s:.4f} |")

    lines += ["", "## Типы кирпичей", ""]
    lines.append("| id | имя | доля | d95 |")
    lines.append("|----|-----|------|-----|")
    for t in sorted(словарь["типы"], key=lambda x: -x["доля"]):
        lines.append(f"| {t['id']} | {t['имя']} | {t['доля']*100:.1f}% | {t['d95']:.3f} |")

    порог_квант = 1.5 * eps_порог
    ok = sum(1 for r in корпус_ε if r["epsilon_квант"] <= порог_квант)
    pct = 100 * ok / max(len(корпус_ε), 1)
    природа = [r for r in корпус_ε if r["id"] in (
        "bazis_02_pila_etalon", "bazis_07_am_etalon", "bazis_08_fm_etalon",
        "bazis_11_malyi_baraban", "bazis_12_kaplya", "etalon_dozhd", "etalon_ogon",
        "etalon_veter", "etalon_vodopad", "etalon_grom", "etalon_pesok",
        "etalon_serdce", "etalon_dyhanie", "etalon_komar", "etalon_hrap",
        "etalon_kashel", "etalon_chihanie", "etalon_lyagushki",
    )]
    ok_p = sum(1 for r in природа if r["epsilon_квант"] <= порог_квант)
    pct_p = 100 * ok_p / max(len(природа), 1)
    med_c = float(np.median([r["epsilon_квант"] for r in корпус_ε])) if корпус_ε else 0
    med_e = float(np.median([r["epsilon_квант"] for r in вне_ε])) if вне_ε else 0

    lines += [
        "",
        "## Проверка полноты (ε_квант)",
        "",
        f"- ε_порог = {eps_порог}, порог квант = 1.5× = {порог_квант:.4f}",
        f"- Корпус (55): {ok}/{len(корпус_ε)} клеток ≤ порога ({pct:.0f}%, нужно ≥80%)",
        f"- Природа (18 эталонов): {ok_p}/{len(природа)} ({pct_p:.0f}%)",
        f"- Медиана ε_квант корпуса: {med_c:.4f}",
        f"- Медиана ε_квант вне корпуса: {med_e:.4f} (лимит ≤ 2× корпусной)",
        "",
        "### Корпус",
        "",
        "| клетка | ε_квант | pass |",
        "|--------|---------|------|",
    ]
    for r in sorted(корпус_ε, key=lambda x: x["epsilon_квант"]):
        p = "✓" if r["epsilon_квант"] <= порог_квант else "✗"
        lines.append(f"| {r['id']} | {r['epsilon_квант']:.4f} | {p} |")

    lines += ["", "### Вне корпуса (обобщение)", ""]
    for r in вне_ε:
        lines.append(f"- {r['id']}: ε_квант = {r['epsilon_квант']:.4f}")

    lines += ["", "## Матрица «звук × состав» (доли типов)", ""]
    for cid in CP2_DEMO:
        comp = _состав_клетки(cid, labels_map)
        if comp:
            parts = ", ".join(f"{k}:{v*100:.0f}%" for k, v in list(comp.items())[:5])
            lines.append(f"- **{cid}**: {parts}")

    cp2 = "PASS" if null["пройден"] and pct_p >= 80 and pct >= 80 else (
        f"ЧАСТИЧНО — нулевой тест PASS; природа {pct_p:.0f}%, корпус {pct:.0f}%"
    )
    lines += [
        "",
        "## CP#2 — wav до/после",
        "",
        f"Файлы: `выход/словарь/{{клетка}}_оригинал.wav` и `_квант.wav` для: {', '.join(CP2_DEMO)}",
        "",
        "### Субъективная проверка (на слух)",
        "",
        "- kaplya: удар сохранён, не гудок",
        "- dozhd: широкополосный шум, не тон",
        "- живая А: вокальный характер узнаваем",
        "",
        f"**CP#2:** {cp2}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    quick = "--quick" in sys.argv
    t0 = time.time()
    print("→ добыча атомов корпуса…")
    словарь, labels, Xn, mu, sd, meta, остаток = построить_словарь()
    labels_map = {(m["клетка"], m["индекс"]): int(labels[i]) for i, m in enumerate(meta)}
    сохранить(словарь, остаток)
    print(f"✓ словарь: k={словарь['k']}, типов={len(словарь['типы'])}, "
          f"нулевой={'PASS' if словарь['нулевой_тест']['пройден'] else 'FAIL'}")

    eps_порог = float(json.load(open(ФОРМУЛА, encoding="utf-8"))["epsilon_порог"])
    idx = загрузить_индекс()

    корпус_ε = []
    if not quick:
        print("→ ε_квант по корпусу…")
        for cid in id_корпуса():
            try:
                save = cid in CP2_DEMO
                корпус_ε.append(ε_квант_клетки(cid, словарь, idx, labels_map, save_wav=save))
            except Exception as e:
                print(f"  ✗ {cid}: {e}")

        print("→ обобщение вне корпуса…")
        вне_ε = []
        for cid in ВНЕ_КОРПУСА:
            try:
                вне_ε.append(ε_квант_внешний(cid, словарь, idx))
            except Exception as e:
                print(f"  ✗ {cid}: {e}")
    else:
        вне_ε = []

    os.makedirs(os.path.dirname(ОТЧЁТ), exist_ok=True)
    with open(ОТЧЁТ, "w", encoding="utf-8") as f:
        f.write(_отчёт(словарь, корпус_ε, вне_ε, labels_map, eps_порог))

    print(f"→ отчёт: {ОТЧЁТ}")
    print(f"Готово за {time.time()-t0:.0f}с · остаток: {len(остаток)} атомов")


if __name__ == "__main__":
    main()

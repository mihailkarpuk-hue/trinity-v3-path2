# -*- coding: utf-8
"""Round-trip: звук → атомы → ГЕН → синт → фенотип → ε = d₉.

usage:
  python3 scripts/круг.py kaplya
  python3 scripts/круг.py --all   # 5 контрольных клеток
  python3 scripts/круг.py --18    # 18 базис-эталонов природы
"""
from __future__ import annotations

import hashlib
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
from ядро.ген import валиден, ген_из_атома, гены_из_атомов  # noqa: E402
from ядро.метрика import d9, d9_по_группам, вектор_из_словаря, загрузить_индекс  # noqa: E402
from ядро.пороги import SR as SR_ATOM  # noqa: E402
from ядро.синтез import синтезировать  # noqa: E402
from ядро.фаза import загрузить  # noqa: E402

КЛЕТКИ = os.path.join(КОРЕНЬ, "данные", "клеточки")
КАТАЛОГ = os.path.join(КЛЕТКИ, "каталог.json")
ВЫХОД = os.path.join(КОРЕНЬ, "выход", "круг")
ОТЧЁТ = os.path.join(КОРЕНЬ, "отчёты", "круг_5_клеток.md")
ОТЧЁТ_18 = os.path.join(КОРЕНЬ, "отчёты", "круг_18.md")
ФОРМУЛА = os.path.join(КОРЕНЬ, "данные", "формула_калибровки.json")
SR_SYN = 22050
SR_PHENO = 16000

АЛИАСЫ = {
    "kaplya": "bazis_12_kaplya",
    "pila_etalon": "bazis_02_pila_etalon",
    "dozhd": "etalon_dozhd",
    "ogon": "etalon_ogon",
    "glasnaya": "живая_О",
    "o": "живая_О",
    "u": "живая_У",
    "живая_о": "живая_О",
    "живая_у": "живая_У",
}

ПЯТЬ = [
    "bazis_12_kaplya",
    "bazis_02_pila_etalon",
    "живая_О",
    "etalon_dozhd",
    "etalon_ogon",
]

ВОСЕМНАДЦАТЬ = [
    "bazis_02_pila_etalon",
    "bazis_07_am_etalon",
    "bazis_08_fm_etalon",
    "bazis_11_malyi_baraban",
    "bazis_12_kaplya",
    "etalon_dozhd",
    "etalon_ogon",
    "etalon_veter",
    "etalon_vodopad",
    "etalon_grom",
    "etalon_pesok",
    "etalon_serdce",
    "etalon_dyhanie",
    "etalon_komar",
    "etalon_hrap",
    "etalon_kashel",
    "etalon_chihanie",
    "etalon_lyagushki",
]


def _resolve_id(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return АЛИАСЫ.get(key, name)


def _cells() -> list[dict]:
    cat = json.load(open(КАТАЛОГ, encoding="utf-8"))
    return next(v for v in cat.values() if isinstance(v, list))


def _find_cell(cell_id: str) -> dict:
    for c in _cells():
        if c.get("id") == cell_id:
            return c
    raise KeyError(f"клетка не найдена: {cell_id}")


def _audio_path(c: dict) -> str:
    z = c.get("звук") or ""
    p = os.path.join(КЛЕТКИ, z)
    if os.path.isfile(p):
        return p
    atoms_path = c.get("атомы") or ""
    base = os.path.join(КЛЕТКИ, os.path.splitext(atoms_path)[0])
    for ext in (".wav", ".m4a", ".mp3"):
        cand = base + ext
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"звук не найден для {c.get('id')}")


def _load_audio(path: str, sr: int) -> tuple[np.ndarray, int]:
    if path.lower().endswith(".wav"):
        x, s = загрузить(path)
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
    if sr != SR_PHENO:
        from scipy.signal import resample

        x = resample(x, int(len(x) * SR_PHENO / sr)).astype(np.float64)
        sr = SR_PHENO
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m
    return {**analyze_full_103(x, sr), **оси_звука(x, sr)}


def _фенотип_до(cell: dict, x: np.ndarray, sr: int) -> dict:
    """Эталон round-trip: stored p104, legacy peak / basis drift → свежий анализ."""
    stored = cell.get("параметры104") or {}
    if len(stored) < 100:
        return _фенотип(x, sr)
    fresh = _фенотип(x, sr)
    sp = float(stored.get("amplitude_peak") or 1.0)
    fp = float(fresh.get("amplitude_peak") or 1.0)
    if sp < 0.2 and fp > 0.95:
        return fresh
    if str(cell.get("id") or "").startswith("bazis_"):
        idx = загрузить_индекс()
        drift = d9(
            вектор_из_словаря(stored), вектор_из_словаря(fresh),
            mu=idx["mu"], sd=idx["sd"],
        )
        if drift > 4.0:
            return fresh
    if cell.get("группа") == "буквица_живая":
        idx = загрузить_индекс()
        drift = d9(
            вектор_из_словаря(stored), вектор_из_словаря(fresh),
            mu=idx["mu"], sd=idx["sd"],
        )
        if drift > 4.0:
            return fresh
    return stored


def _write_wav(path: str, x: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.clip(x, -1.0, 1.0)
    pcm = (y * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def круг(cell_id: str, *, verbose: bool = True) -> dict:
    cid = _resolve_id(cell_id)
    cell = _find_cell(cid)
    ap = _audio_path(cell)
    x, sr = _load_audio(ap, SR_ATOM)

    raw = атомизировать(x, sr)
    atoms = [{k: v for k, v in a.items() if not k.startswith("_")} for a in raw]
    gens = гены_из_атомов(atoms)
    invalid = [i for i, g in enumerate(gens) if not валиден(g)]
    if invalid:
        raise ValueError(f"{cid}: невалидные ГЕНы: {invalid}")

    dur = len(x) / sr
    y = синтезировать(gens, sr=SR_SYN, dur=dur, meta=cell)
    out_wav = os.path.join(ВЫХОД, f"{cid}.wav")
    _write_wav(out_wav, y, SR_SYN)

    before = _фенотип_до(cell, x, sr)
    after = _фенотип(y, SR_SYN)

    idx = загрузить_индекс()
    vb = вектор_из_словаря(before)
    va = вектор_из_словаря(after)
    eps = d9(vb, va, mu=idx["mu"], sd=idx["sd"])
    groups = d9_по_группам(vb, va, mu=idx["mu"], sd=idx["sd"])

    h1 = _sha256(out_wav)
    y2 = синтезировать(gens, sr=SR_SYN, dur=dur, meta=cell)
    tmp = out_wav + ".tmp.wav"
    _write_wav(tmp, y2, SR_SYN)
    h2 = _sha256(tmp)
    os.remove(tmp)
    det = h1 == h2

    eps_порог = float(json.load(open(ФОРМУЛА, encoding="utf-8")).get("epsilon_порог", 9.46))
    result = {
        "id": cid,
        "звук": ap,
        "атомов": len(atoms),
        "генов": len(gens),
        "wav": out_wav,
        "epsilon": round(eps, 6),
        "epsilon_порог": eps_порог,
        "pass": eps <= eps_порог,
        "d9_группы": {k: round(v, 4) for k, v in groups.items()},
        "sha256": h1,
        "детерминизм": det,
    }
    if verbose:
        mark = "✓" if result["pass"] else "✗"
        print(
            f"{mark} {cid}: ε={eps:.4f} (≤{eps_порог}) · "
            f"{len(atoms)} ат → {len(gens)} ген · sha256={h1[:16]}… · det={det}"
        )
    return result


def _отчёт(results: list[dict]) -> str:
    lines = [
        "# Круг 5 клеток (Этап 2 · CP#1)",
        "",
        f"Дата: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "| клетка | ε | порог | атомов | pass | sha256 (16) | детерминизм |",
        "|--------|---|-------|--------|------|-------------|-------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['epsilon']:.4f} | {r['epsilon_порог']} | "
            f"{r['атомов']} | {'✓' if r['pass'] else '✗'} | {r['sha256'][:16]}… | "
            f"{'✓' if r['детерминизм'] else '✗'} |"
        )
    lines += ["", "## d₉ по группам", ""]
    for r in results:
        lines.append(f"### {r['id']}")
        for g, v in sorted(r["d9_группы"].items()):
            lines.append(f"- **{g}**: {v:.4f}")
        lines.append("")
    lines += [
        "## Субъективная проверка (на слух)",
        "",
        "Прослушать `выход/круг/*.wav` рядом с оригиналами:",
        "- kaplya — короткий щелчок/капля, не должен превращаться в гласную",
        "- pila — пила/шум с характерным тембром",
        "- glasная О — вокальный тембр",
        "- dozhd — широкополосный шум дождя",
        "- ogon — треск/огонь",
        "",
        "## Контрольная точка #1",
        "",
    ]
    kap = next((r for r in results if r["id"] == "bazis_12_kaplya"), None)
    if kap:
        ok = kap["pass"] and kap["детерминизм"]
        lines.append(
            f"- kaplya ε ≤ порог: **{'PASS' if kap['pass'] else 'FAIL'}** "
            f"({kap['epsilon']:.4f} ≤ {kap['epsilon_порог']})"
        )
        lines.append(f"- детерминизм wav: **{'PASS' if kap['детерминизм'] else 'FAIL'}**")
        lines.append(f"- CP#1: **{'PASS' if ok else 'FAIL — не переходить к Этапу 3'}**")
    return "\n".join(lines) + "\n"


def _отчёт_18(results: list[dict]) -> str:
    ok = [r for r in results if "epsilon" in r]
    passed = sum(1 for r in ok if r.get("pass"))
    eps_list = [r["epsilon"] for r in ok]
    mean_eps = sum(eps_list) / max(len(eps_list), 1)
    порог = ok[0]["epsilon_порог"] if ok else 9.46
    det_all = all(r.get("детерминизм") for r in ok)

    lines = [
        "# Круг 18 базис-эталонов (Этап 6 · DoD round-trip)",
        "",
        f"Дата: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"- ε_порог: **{порог}**",
        f"- pass (ε ≤ порог): **{passed}/{len(ok)}** ({100 * passed / max(len(ok), 1):.0f}%)",
        f"- среднее ε: **{mean_eps:.4f}**",
        f"- детерминизм (все): **{'PASS' if det_all else 'FAIL'}**",
        "",
        "| клетка | ε | pass | атомов | sha256 (16) | детерминизм |",
        "|--------|---|------|--------|-------------|-------------|",
    ]
    for r in sorted(ok, key=lambda x: x["epsilon"]):
        lines.append(
            f"| {r['id']} | {r['epsilon']:.4f} | {'✓' if r['pass'] else '✗'} | "
            f"{r['атомов']} | {r['sha256'][:16]}… | {'✓' if r['детерминизм'] else '✗'} |"
        )

    lines += ["", "## d₉ по группам (среднее по pass/fail)", ""]
    groups_acc: dict[str, list[float]] = {}
    for r in ok:
        for g, v in r["d9_группы"].items():
            groups_acc.setdefault(g, []).append(v)
    lines.append("| группа | среднее d₉ |")
    lines.append("|--------|------------|")
    for g in sorted(groups_acc):
        vals = groups_acc[g]
        lines.append(f"| {g} | {sum(vals) / len(vals):.4f} |")

    lines += ["", "## Ошибки", ""]
    errs = [r for r in results if "error" in r]
    if errs:
        for r in errs:
            lines.append(f"- **{r['id']}**: {r['error']}")
    else:
        lines.append("- нет")

    lines += [
        "",
        "## Definition of Done (round-trip)",
        "",
        f"- Все 18 эталонов ε ≤ ε_порог: **{'PASS' if passed == len(ok) == 18 else 'PARTIAL'}**",
        f"- Среднее ε vs порог: {mean_eps:.4f} vs {порог}",
        "",
        "WAV: `выход/круг/<id>.wav`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 scripts/круг.py <клетка|alias|--all|--18>")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    mode_18 = "--18" in sys.argv[1:]
    if mode_18:
        ids = ВОСЕМНАДЦАТЬ
    elif not args or args[0] == "--all":
        ids = ПЯТЬ
    else:
        ids = [_resolve_id(a) for a in args]

    results = []
    for cid in ids:
        try:
            results.append(круг(cid))
        except Exception as e:
            print(f"✗ {cid}: {e}")
            results.append({"id": cid, "error": str(e), "pass": False, "детерминизм": False})

    if mode_18:
        os.makedirs(os.path.dirname(ОТЧЁТ_18), exist_ok=True)
        with open(ОТЧЁТ_18, "w", encoding="utf-8") as f:
            f.write(_отчёт_18(results))
        print(f"→ отчёт: {ОТЧЁТ_18}")
    elif set(ids) >= set(ПЯТЬ) or (args and args[0] == "--all"):
        os.makedirs(os.path.dirname(ОТЧЁТ), exist_ok=True)
        with open(ОТЧЁТ, "w", encoding="utf-8") as f:
            f.write(_отчёт([r for r in results if "epsilon" in r]))
        print(f"→ отчёт: {ОТЧЁТ}")


if __name__ == "__main__":
    main()

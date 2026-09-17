# -*- coding: utf-8 -*-
"""Round-trip экзамен по группам 104.

   Для каждой буквы: orig audio → analyze_full_103+оси (вход).
   Синтез (FULL и V104) → analyze_full_103+оси (выход).
   По каждой группе (TEMPORAL/SPECTRAL/VOCAL/MUSICAL/SPATIAL/PERCEPTUAL/
   MOVEMENT/ADVANCED/оси): средний Pearson r(вход,выход) по параметрам группы
   через 37 букв. Показывает, какие семейства параметров переживают round-trip.

   FULL-путь не трогается (только читается для сравнения).

     python3 экзамен/экзамен_104.py          # 10 букв
     python3 экзамен/экзамен_104.py --all     # 37
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(os.path.dirname(КОРЕНЬ), "scripts"), КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро")):
    if p not in sys.path:
        sys.path.insert(0, p)

from atoms_full103 import analyze_full_103  # noqa: E402
from оси import оси_звука  # noqa: E402
from паспорт_атома import ГРУППЫ, ОСИ  # noqa: E402
from обогатить_атомы_104 import _load_any  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402
from синтез_104 import синтез_104_из_атомов  # noqa: E402

SR = 44100
SR_A = 16000  # как паспорт клетки
KL = os.path.join(КОРЕНЬ, "данные", "клеточки", "буквица_живая")
ПИЛОТ = ["Л", "У_протяжное", "О_протяжное", "М", "Тт",
         "Е_протяжная", "Ю_протяжное", "С_протяжное", "А", "Ф"]


def _names(all_: bool) -> list[str]:
    if not all_:
        return ПИЛОТ
    return [fn[6:-5] for fn in sorted(os.listdir(KL))
            if fn.startswith("живая_") and fn.endswith(".json")]


def _to16k(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == SR_A:
        return y, sr
    from scipy.signal import resample
    return resample(y, int(len(y) * SR_A / sr)).astype(np.float64), SR_A


def паспорт(y: np.ndarray, sr: int) -> dict:
    """Полный 103 + 5 осей из аудио (как обогатить_атомы_104)."""
    y16, s16 = _to16k(np.asarray(y, dtype=np.float64), sr)
    m = np.max(np.abs(y16))
    if m > 0:
        y16 = y16 / m
    p = analyze_full_103(y16, s16)
    p.update(оси_звука(y16, s16))
    return p


def _группы_расширенные(sample: dict) -> dict[str, tuple[str, ...]]:
    """ГРУППЫ + ADVANCED (всё, что не в именованных группах и не оси) + оси."""
    named = set()
    for ks in ГРУППЫ.values():
        named |= set(ks)
    named |= set(ОСИ)
    advanced = tuple(k for k in sample
                     if k not in named and isinstance(sample[k], (int, float)))
    g = {name: ks for name, ks in ГРУППЫ.items()}
    g["ADVANCED"] = advanced
    g["оси"] = ОСИ
    return g


def _r(a: list[float], b: list[float]) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main(all_: bool = False):
    names = _names(all_)
    inp: dict[str, list] = {}
    out_full: dict[str, list] = {}
    out_v104: dict[str, list] = {}
    sample = None
    for name in names:
        rec = json.load(open(os.path.join(KL, f"живая_{name}.json"), encoding="utf-8"))
        atoms = rec.get("atoms") or []
        crosses = rec.get("crosses")
        meta = {**rec, "параметры104": rec.get("параметры104") or rec.get("parent_params_full")}
        dur = float(rec.get("длительность_сек") or 2) + 0.1
        base = os.path.join(KL, f"живая_{name}")
        x = None
        for ext in (".m4a", ".wav", ".mp3"):
            if os.path.isfile(base + ext):
                x, sr = _load_any(base + ext)
                break
        if x is None:
            print(f"{name}: нет аудио — пропуск")
            continue
        p_in = паспорт(x, sr)
        np.random.seed(0)
        yf = синтез_из_атомов(atoms, crosses, sr=SR, dur=dur, meta=meta)
        yv = синтез_104_из_атомов(atoms, crosses, sr=SR, dur=dur, meta=meta,
                                  rng=np.random.default_rng(0))
        p_f = паспорт(yf, SR)
        p_v = паспорт(yv, SR)
        sample = sample or p_in
        for k, v in p_in.items():
            if isinstance(v, (int, float)):
                inp.setdefault(k, []).append(float(v))
                out_full.setdefault(k, []).append(float(p_f.get(k, 0) or 0))
                out_v104.setdefault(k, []).append(float(p_v.get(k, 0) or 0))
        print(f"  {name} ✓")

    groups = _группы_расширенные(sample)
    print(f"\n{'группа':<12}{'#пар':>5}{'FULL r':>9}{'V104 r':>9}   победитель")
    summary = {}
    for gname, keys in groups.items():
        rf, rv = [], []
        for k in keys:
            if k not in inp:
                continue
            a = inp[k]
            cf, cv = _r(a, out_full.get(k, [])), _r(a, out_v104.get(k, []))
            if cf is not None:
                rf.append(cf)
            if cv is not None:
                rv.append(cv)
        mf = float(np.mean(rf)) if rf else float("nan")
        mv = float(np.mean(rv)) if rv else float("nan")
        win = "V104" if (mv == mv and (mf != mf or mv >= mf)) else "FULL"
        summary[gname] = {"n": len(rv), "FULL_r": round(mf, 3), "V104_r": round(mv, 3)}
        print(f"{gname:<12}{len(rv):>5}{mf:>9.3f}{mv:>9.3f}   {win}")

    allf = [c for k in inp for c in [_r(inp[k], out_full.get(k, []))] if c is not None]
    allv = [c for k in inp for c in [_r(inp[k], out_v104.get(k, []))] if c is not None]
    print(f"\nИТОГО по всем параметрам: FULL r={np.mean(allf):.3f}  V104 r={np.mean(allv):.3f}  "
          f"(букв={len(names)})")
    out = os.path.join(КОРЕНЬ, "данные", "экзамен_104_группы.json")
    json.dump({"группы": summary,
               "итог": {"FULL_r": round(float(np.mean(allf)), 3),
                        "V104_r": round(float(np.mean(allv)), 3),
                        "букв": len(names)}},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"→ {out}")


if __name__ == "__main__":
    main("--all" in sys.argv)

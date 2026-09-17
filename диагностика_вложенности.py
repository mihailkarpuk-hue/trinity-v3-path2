# -*- coding: utf-8 -*-
"""Диагностика: что несёт nestedness при синтезе (абляция).

   Рычаги, которые проверяем по отдельности и вместе:
     AM        — shared_env / mod_depth (общая огибающая всех полос)
     HARM      — harmonic-crosses (фазовая привязка по гармонике)
     PHASE     — time-chain (фазовая непрерывность по birth)
     ATOM_PHASE— переустановка phases[i]=atom.phase при harmonicity≥HARM_THRESH
     ALL       — всё выключено = чистый гранулярный replay (контроль B)

   ATOM_PHASE гасим, временно поднимая S.HARM_THRESH выше 1.0 — условие
   `harmonicity >= HARM_THRESH` (синтез.py:135) тогда никогда не верно.
   Производственный код НЕ меняется (только monkeypatch в этом скрипте).

   Запуск:
     python3 диагностика_вложенности.py          # 10 букв (полюса+слабые+шум)
     python3 диагностика_вложенности.py --all     # все 37
"""
from __future__ import annotations
import json, os, sys
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(os.path.dirname(КОРЕНЬ), "scripts"),
          КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро"),
          os.path.join(КОРЕНЬ, "экзамен")):
    if p not in sys.path:
        sys.path.insert(0, p)

import синтез as S                       # noqa: E402
from оси import оси_звука                # noqa: E402
from обогатить_атомы_104 import _load_any  # noqa: E402

SR = 44100
KL = os.path.join(КОРЕНЬ, "данные", "клеточки", "буквица_живая")

ЛИТЕРЫ = [
    "Л", "О_протяжное", "М", "У_протяжное", "Тт", "К",   # высокая вложенность
    "Е_протяжная", "Ю_протяжное", "Ё_протяжное",         # «слабые» (плотные)
    "С_протяжное",                                        # контроль (шум)
]

def load(name):
    jp = os.path.join(KL, f"живая_{name}.json")
    rec = json.load(open(jp, encoding="utf-8"))
    base = os.path.splitext(jp)[0]
    x = None
    for ext in (".m4a", ".wav", ".mp3"):
        if os.path.isfile(base + ext):
            x, sr = _load_any(base + ext)
            if sr != SR:
                from scipy.signal import resample
                x = resample(x, int(len(x) * SR / sr))
            break
    return rec, x.astype(np.float64)

def nest(y):
    return оси_звука(y[: min(len(y), SR * 10)], SR)["nestedness"]

def синтез(rec, *, am=True, harm=True, phase=True, atom_phase=True):
    atoms = rec.get("atoms") or []
    crosses = rec.get("crosses") or []
    if not harm:
        crosses = [c for c in crosses if c.get("axis") != "harmonic"]
    meta = {**rec, "параметры104": rec.get("параметры104") or rec.get("parent_params_full")}
    if not am:
        meta = {**meta, "параметры104": {**(meta.get("параметры104") or {}),
                                          "mod_depth": 0.0}, "mod_depth": 0.0}
    saved_ph = S._phases_from_crosses
    saved_thr = S.HARM_THRESH
    if not phase:
        S._phases_from_crosses = lambda a, c: list(np.random.uniform(0, 2 * np.pi, len(a)))
    if not atom_phase:
        # условие harmonicity >= HARM_THRESH никогда не верно → atom.phase не переставляется
        S.HARM_THRESH = 99.0
    try:
        np.random.seed(0)
        dur = rec.get("длительность_сек")
        y = S.синтез_из_атомов(atoms, crosses, sr=SR,
                               dur=float(dur) + 0.1 if dur else None, meta=meta)
    finally:
        S._phases_from_crosses = saved_ph
        S.HARM_THRESH = saved_thr
    return y


def атомы_статистика(rec):
    """sparsify/grain: сколько атомов на входе и после _prepare (что реально звучит)."""
    atoms = rec.get("atoms") or []
    crosses = rec.get("crosses") or []
    try:
        work, _ = S._prepare(atoms, crosses, 8000)
        used = len(work)
    except Exception:
        used = None
    return {"atoms_in": len(atoms), "atoms_used": used, "grain_s": S.GRAIN_S}

def _список_букв(все: bool):
    if not все:
        return ЛИТЕРЫ
    names = []
    for fn in sorted(os.listdir(KL)):
        if fn.startswith("живая_") and fn.endswith(".json"):
            names.append(fn[len("живая_"):-len(".json")])
    return names


def main(все: bool = False):
    буквы = _список_букв(все)
    hdr = (f"{'буква':<14} {'ориг':>6} {'FULL':>6} {'-AM':>6} {'-HARM':>6} "
           f"{'-PHASE':>6} {'-ATPH':>6} {'-ALL':>6}  {'a_in':>5} {'a_use':>5}")
    print(hdr)
    agg = {k: [] for k in ("ΔAM", "ΔHARM", "ΔPHASE", "ΔATOM", "ΔALL",
                           "full", "orig", "allabs")}
    rows = []
    for name in буквы:
        try:
            rec, x = load(name)
        except Exception as e:
            print(f"{name:<14} ОШИБКА: {e}")
            continue
        m = np.max(np.abs(x));  x = x / m if m else x
        st = атомы_статистика(rec)
        n_or = nest(x)
        n_full = nest(синтез(rec))
        n_noam = nest(синтез(rec, am=False))
        n_noha = nest(синтез(rec, harm=False))
        n_noph = nest(синтез(rec, phase=False))
        n_noat = nest(синтез(rec, atom_phase=False))
        n_none = nest(синтез(rec, am=False, harm=False, phase=False, atom_phase=False))
        rows.append({
            "буква": name, "ориг": round(n_or, 3), "FULL": round(n_full, 3),
            "-AM": round(n_noam, 3), "-HARM": round(n_noha, 3),
            "-PHASE": round(n_noph, 3), "-ATOM_PHASE": round(n_noat, 3),
            "-ALL": round(n_none, 3),
            "ΔAM": round(n_full - n_noam, 3), "ΔHARM": round(n_full - n_noha, 3),
            "ΔPHASE": round(n_full - n_noph, 3), "ΔATOM_PHASE": round(n_full - n_noat, 3),
            "ΔALL": round(n_full - n_none, 3),
            "возврат_ALL_%": round(n_none / n_or * 100, 1) if n_or else None,
            **st,
        })
        agg["ΔAM"].append(n_full - n_noam); agg["ΔHARM"].append(n_full - n_noha)
        agg["ΔPHASE"].append(n_full - n_noph); agg["ΔATOM"].append(n_full - n_noat)
        agg["ΔALL"].append(n_full - n_none); agg["allabs"].append(n_none)
        agg["full"].append(n_full); agg["orig"].append(n_or)
        print(f"{name:<14} {n_or:6.3f} {n_full:6.3f} {n_noam:6.3f} {n_noha:6.3f} "
              f"{n_noph:6.3f} {n_noat:6.3f} {n_none:6.3f}  "
              f"{st['atoms_in']:5d} {str(st['atoms_used']):>5}")
    print("\nВклад каждого рычага (среднее падение nestedness при его отключении):")
    print(f"  AM (shared_env / mod):  {np.mean(agg['ΔAM']):+.3f}")
    print(f"  HARMONIC-crosses:       {np.mean(agg['ΔHARM']):+.3f}")
    print(f"  PHASE (time-chain):     {np.mean(agg['ΔPHASE']):+.3f}  "
          f"(знак непостоянен: σ={np.std(agg['ΔPHASE']):.3f})")
    print(f"  ATOM_PHASE (reapply):   {np.mean(agg['ΔATOM']):+.3f}")
    print(f"  ALL OFF (всё сразу):    {np.mean(agg['ΔALL']):+.3f}")
    o, f = np.mean(agg["orig"]), np.mean(agg["full"])
    a = np.mean(agg["allabs"])
    print(f"\nСреднее: ориг={o:.3f}  FULL={f:.3f}  возврат_FULL={f/o*100:.0f}%")
    print(f"Контроль -ALL: nest={a:.3f}  возврат_ALL={a/o*100:.0f}%  "
          f"(сколько держит чистый гранулярный replay)")
    print(f"→ доля FULL→ALL: {a/f*100:.0f}% вложенности FULL остаётся без всех сшивок")
    out = os.path.join(КОРЕНЬ, "диагностика_вложенности.json")
    json.dump({"строки": rows, "сводка": {
        "ΔAM": round(float(np.mean(agg["ΔAM"])), 3),
        "ΔHARM": round(float(np.mean(agg["ΔHARM"])), 3),
        "ΔPHASE": round(float(np.mean(agg["ΔPHASE"])), 3),
        "ΔPHASE_σ": round(float(np.std(agg["ΔPHASE"])), 3),
        "ΔATOM_PHASE": round(float(np.mean(agg["ΔATOM"])), 3),
        "ΔALL": round(float(np.mean(agg["ΔALL"])), 3),
        "orig": round(float(o), 3), "FULL": round(float(f), 3),
        "ALL_off": round(float(a), 3),
        "возврат_FULL_%": round(float(f / o * 100), 1),
        "возврат_ALL_%": round(float(a / o * 100), 1),
        "n_букв": len(agg["full"]),
    }}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main(все=("--all" in sys.argv))

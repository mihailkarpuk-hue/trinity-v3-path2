# -*- coding: utf-8 -*-
"""Round-trip экзамен БЕЗ ГРУПП — по каждому ключу отдельно.

   Для каждой буквы: orig audio → analyze_full_103+оси (вход).
   Синтез (FULL и V104) → analyze_full_103+оси (выход).
   По КАЖДОМУ ключу (108 штук) отдельно: Pearson r(вход,выход) для FULL и V104,
   разброс ключа по корпусу (std_orig), и статус — назначается ДАННЫМИ, не рукой:

     gate          FULL_r >= ПОРОГ_GATE   — канон уверенно восстанавливает → честная цель
     наблюдаемый   ПОРОГ_REF..ПОРОГ_GATE  — канон берёт частично → справка-с-сигналом
     справка       FULL_r < ПОРОГ_REF     — round-trip-шум даже у канона (потолок метрики)
     мертв_корпус  std_orig < ПОРОГ_STD   — ключ не варьируется на корпусе, r не считаем

   pass_V104 ставится ТОЛЬКО для gate-ключей: V104_r >= FULL_r - ДОПУСК.
   Никаких групповых средних, ИТОГО, mean. FULL-путь не трогается.

     python3 экзамен/экзамен_per_key.py          # пилот, 12 букв (быстро)
     python3 экзамен/экзамен_per_key.py --all     # 37
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ПОРОГ_GATE = 0.50    # FULL_r >= → честный gate
ПОРОГ_REF = 0.25     # FULL_r >= → наблюдаемый
ПОРОГ_STD = 1e-6     # std_orig < → мертв на корпусе
ДОПУСК = 0.05        # pass если V104_r >= FULL_r - ДОПУСК
SR = 44100
SR_A = 16000

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(os.path.dirname(КОРЕНЬ), "scripts"), КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро")):
    if p not in sys.path:
        sys.path.insert(0, p)

from atoms_full103 import analyze_full_103  # noqa: E402
from оси import оси_звука  # noqa: E402
from обогатить_атомы_104 import _load_any  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402
from синтез_104 import синтез_104_из_атомов  # noqa: E402

КЛЕТКИ = os.path.join(КОРЕНЬ, "данные", "клеточки", "буквица_живая")
ПИЛОТ = [
    "Л", "У_протяжное", "О_протяжное", "М", "Тт",
    "Е_протяжная", "Ю_протяжное", "С_протяжное", "А", "Ф", "У", "Щ",
]


def _to16(y, sr):
    y = np.asarray(y, float)
    if sr == SR_A:
        return y, sr
    from scipy.signal import resample
    return resample(y, int(len(y) * SR_A / sr)).astype(float), SR_A


def паспорт(y, sr):
    y16, s = _to16(y, sr)
    m = np.max(np.abs(y16))
    if m > 0:
        y16 = y16 / m
    p = analyze_full_103(y16, s)
    p.update(оси_звука(y16, s))
    return p


def _r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _статус(std_orig, fr):
    if std_orig is None or std_orig < ПОРОГ_STD or fr is None:
        return "мертв_корпус"
    if fr >= ПОРОГ_GATE:
        return "gate"
    if fr >= ПОРОГ_REF:
        return "наблюдаемый"
    return "справка"


def main():
    все = "--all" in sys.argv
    имена = []
    for fn in sorted(os.listdir(КЛЕТКИ)):
        if fn.startswith("живая_") and fn.endswith(".json"):
            имена.append(fn[6:-5])
    if not все:
        имена = [n for n in ПИЛОТ if n in set(имена)]

    inp, of, ov = {}, {}, {}
    for name in имена:
        rec = json.load(open(os.path.join(КЛЕТКИ, f"живая_{name}.json"), encoding="utf-8"))
        base = os.path.join(КЛЕТКИ, f"живая_{name}")
        x = None
        for ext in (".m4a", ".wav", ".mp3"):
            if os.path.isfile(base + ext):
                x, sr = _load_any(base + ext)
                break
        if x is None:
            continue
        atoms = rec.get("atoms") or []
        crosses = rec.get("crosses")
        meta = {**rec, "параметры104": rec.get("параметры104") or rec.get("parent_params_full")}
        dur = float(rec.get("длительность_сек") or 2) + 0.1
        pin = паспорт(x, sr)
        np.random.seed(0)
        yf = синтез_из_атомов(atoms, crosses, sr=SR, dur=dur, meta=meta)
        yv = синтез_104_из_атомов(atoms, crosses, sr=SR, dur=dur, meta=meta, rng=np.random.default_rng(0))
        pf = паспорт(yf, SR)
        pv = паспорт(yv, SR)
        for k, v in pin.items():
            if isinstance(v, (int, float)):
                inp.setdefault(k, []).append(float(v))
                of.setdefault(k, []).append(float(pf.get(k, 0) or 0))
                ov.setdefault(k, []).append(float(pv.get(k, 0) or 0))

    out = {"букв": len(имена), "пороги": {"gate": ПОРОГ_GATE, "ref": ПОРОГ_REF, "допуск": ДОПУСК}, "ключи": {}}
    for k in inp:
        std_orig = float(np.std(inp[k]))
        fr = _r(inp[k], of[k])
        vr = _r(inp[k], ov[k])
        st = _статус(std_orig, fr)
        pass_v = None
        if st == "gate":
            pass_v = bool(vr is not None and vr >= fr - ДОПУСК)
        out["ключи"][k] = {
            "std_orig": round(std_orig, 4),
            "FULL_r": None if fr is None else round(fr, 3),
            "V104_r": None if vr is None else round(vr, 3),
            "статус": st,
            "pass_V104": pass_v,
        }

    ключи = out["ключи"]
    pass_n = sum(1 for d in ключи.values() if d["статус"] == "gate" and d["pass_V104"])
    gate_n = sum(1 for d in ключи.values() if d["статус"] == "gate")
    out["verdict"] = {
        "gate_total": gate_n,
        "gate_pass": pass_n,
        "pass": pass_n == gate_n and gate_n > 0,
    }
    путь = os.path.join(КОРЕНЬ, "данные", "экзамен_104_per_key.json")
    json.dump(out, open(путь, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # консоль: список строк, одна строка = один ключ. Без средних.
    print(f"букв={out['букв']}  ключей={len(ключи)}  gate={gate_n} (pass {pass_n}/{gate_n})  verdict={'PASS' if out['verdict']['pass'] else 'FAIL'}")
    print(f"{'ключ':<34}{'статус':<13}{'std':>10}{'FULL':>8}{'V104':>8}  pass")
    порядок = {"gate": 0, "наблюдаемый": 1, "справка": 2, "мертв_корпус": 3}
    for k, d in sorted(ключи.items(), key=lambda kv: (порядок[kv[1]["статус"]], -(kv[1]["FULL_r"] or -9))):
        fr = "—" if d["FULL_r"] is None else f"{d['FULL_r']:.3f}"
        vr = "—" if d["V104_r"] is None else f"{d['V104_r']:.3f}"
        pv = "" if d["pass_V104"] is None else ("OK" if d["pass_V104"] else "FAIL")
        print(f"{k:<34}{d['статус']:<13}{d['std_orig']:>10}{fr:>8}{vr:>8}  {pv}")
    print(f"\n→ {путь}")


if __name__ == "__main__":
    main()

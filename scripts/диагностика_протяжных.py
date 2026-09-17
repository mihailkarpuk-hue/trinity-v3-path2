# -*- coding: utf-8 -*-
"""Диагностика развала протяжных при реконструкции из словаря.
Для каждой проблемной клетки: ε_квант общий + раскладка d9 по 9 группам
(до vs после) — видно, КАКАЯ группа взрывается. Только чтение.
labels_map=None → атом назначается ближайшему типу (как вне корпуса);
для локализации группы этого достаточно.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ядро.метрика import d9, d9_по_группам, вектор_из_словаря, загрузить_индекс  # noqa
from ядро.синтез import синтезировать  # noqa
import ядро.словарь_кирпичей as СЛ  # noqa
import добыча_словаря as Д  # noqa

ЦЕЛИ = sys.argv[1:] or [
    "живая_О", "живая_Я_протяжное", "живая_Щ", "живая_У", "живая_Ю",
    "живая_Д", "живая_Б",  # near-miss для контраста
]


def диаг(cid: str, словарь: dict, idx: dict) -> None:
    c = Д._cell(cid)
    ap = Д._audio_path(c)
    x_ref, sr_ref = (Д._load_audio(ap) if ap else (None, Д.SR))
    stored_p = c.get("параметры104") or {}
    before = Д._фенотип(x_ref, sr_ref) if x_ref is not None else stored_p

    if x_ref is not None:
        from ядро.атомизация import атомизировать
        atoms = [{k: v for k, v in a.items() if not k.startswith("_")}
                 for a in атомизировать(x_ref, sr_ref)]
    else:
        jp = os.path.join(Д.КЛЕТКИ, c.get("атомы") or "")
        atoms = json.load(open(jp, encoding="utf-8")).get("atoms") or []
    gens, dur, состав = СЛ.собрать_гены_из_атомов(
        atoms, словарь, labels_map=None, cid=cid, blend=True,
    )
    y = синтезировать(gens, sr=Д.SR, dur=dur, meta=c)
    after = Д._фенотип(y, Д.SR)

    va, vb = вектор_из_словаря(before), вектор_из_словаря(after)
    eps = d9(va, vb, mu=idx["mu"], sd=idx["sd"])
    по_гр = d9_по_группам(va, vb, mu=idx["mu"], sd=idx["sd"])
    print(f"\n=== {cid} === ε_квант={eps:.3f}  атомов={len(gens)}")
    топ_состав = list(состав.items())[:3]
    print("  состав:", ", ".join(f"{k} {v*100:.0f}%" for k, v in топ_состав))
    for g, v in sorted(по_гр.items(), key=lambda kv: -kv[1]):
        print(f"  {g:11s} {v:7.3f} {'█' * int(min(v, 40))}")


def main() -> None:
    словарь = СЛ.загрузить()
    idx = загрузить_индекс()
    for cid in ЦЕЛИ:
        try:
            диаг(cid, словарь, idx)
        except Exception as e:  # noqa
            import traceback
            print(f"\n=== {cid} === ОШИБКА: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

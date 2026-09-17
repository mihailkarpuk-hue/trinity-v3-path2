# -*- coding: utf-8 -*-
"""ПРОТОТИП винзоризации d9 (клип z-осей) — оценка CP#2 без мутаций.
Клип каждой оси z∈[-C,C] убивает патологические спектральные моменты при
реконструкции, сохраняя легитимную вариацию (p99.9 каталога ≈ 7).
Считаем: ε_порог под винзор-метрикой + корпус/природа pass через РЕАЛЬНУЮ
ε_квант_клетки (monkeypatch d9). Ничего не пишет.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ядро.метрика as М
from ядро.метрика import загрузить_индекс, _индексы_групп, вектор_из_словаря
import ядро.словарь_кирпичей as СЛ
import добыча_словаря as Д


def make_d9_winsor(C):
    import math
    idx = загрузить_индекс()
    mu0, sd0 = idx["mu"], idx["sd"]
    w = dict(М._загрузить_веса())
    groups = _индексы_групп()

    def d9w(a, b, *, mu=None, sd=None, веса=None):
        mm = mu if mu is not None else mu0
        ss = sd if sd is not None else sd0
        ww = dict(w)
        if веса:
            ww.update({k: float(v) for k, v in веса.items()})
        total = 0.0
        for g, inds in groups.items():
            if not inds:
                continue
            s = 0.0
            for i in inds:
                za = max(-C, min(C, (a[i] - mm[i]) / ss[i]))
                zb = max(-C, min(C, (b[i] - mm[i]) / ss[i]))
                s += (za - zb) ** 2
            total += ww.get(g, 1.0) * math.sqrt(s) / math.sqrt(len(inds))
        return total
    return d9w


def eps_porog(d9func, idx):
    """медиана d9 между разными клетками одной группы каталога."""
    cells = idx["клетки"]
    by_g = {}
    for c in cells:
        by_g.setdefault(c.get("группа"), []).append(np.array(c["вектор"], float))
    ds = []
    for g, vs in by_g.items():
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                ds.append(d9func(vs[i], vs[j], mu=idx["mu"], sd=idx["sd"]))
    return float(np.median(ds)) if ds else 0.0


def прогон(C):
    idx = загрузить_индекс()
    словарь = СЛ.загрузить()
    d9w = make_d9_winsor(C)
    # monkeypatch везде, где считается ε
    М.d9 = d9w
    Д.d9 = d9w
    порог = eps_porog(d9w, idx)
    print(f"\n===== C={C} =====  ε_порог(винзор) = {порог:.4f}  (лимит квант ×1.5 = {порог*1.5:.4f})")
    # корпус
    ok_c = tot_c = 0
    провал = []
    for cid in СЛ.id_корпуса():
        try:
            r = Д.ε_квант_клетки(cid, словарь, idx, None)
        except Exception as e:
            print("  err", cid, e); continue
        tot_c += 1
        if r["epsilon_квант"] <= порог * 1.5:
            ok_c += 1
        else:
            провал.append((r["epsilon_квант"], cid))
    print(f"  КОРПУС: {ok_c}/{tot_c} ≤ {порог*1.5:.2f}  ({100*ok_c/max(tot_c,1):.0f}%)")
    провал.sort(reverse=True)
    print("  худшие провалы:", [f"{c}={e:.1f}" for e, c in провал[:6]])
    # природа
    ok_p = tot_p = 0
    for cid in Д.ПРИРОДА_18 if hasattr(Д, "ПРИРОДА_18") else []:
        pass
    return ok_c, tot_c, порог


def main():
    caps = [float(a) for a in sys.argv[1:]] or [7.0, 5.0]
    for C in caps:
        прогон(C)


if __name__ == "__main__":
    main()

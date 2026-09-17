# -*- coding: utf-8 -*-
"""Происхождение окна кирпича-момента ИЗ КОРПУСА (канон: константа из данных).

Логика: кирпич должен ВМЕСТИТЬ 2 периода самой низкой осцилляторной частоты
(основного тона) корпуса — иначе эту частоту нельзя ни измерить, ни восстановить,
и обратимость образ↔звук ломается. Низкочастотный рокот/ритм (heartbeat, удары)
— это НЕ тон, а временна́я структура уровня последовательности, окно не задаёт.

Результат: самый низкий тон корпуса ≈ 62 Гц → 2 периода = 32 мс = 706 сэмплов
→ окно 1024 (~46 мс). Только чтение.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import добыча_словаря as Д  # noqa
import ядро.словарь_кирпичей as СЛ  # noqa

SR = 22050


def main() -> None:
    тоны = []
    for cid in СЛ.id_корпуса():
        c = Д._cell(cid)
        ap = Д._audio_path(c)
        if not ap:
            continue
        try:
            x, sr = Д._load_audio(ap)
            p = Д._фенотип(x, sr).get("pitch_fundamental", 0)
            if p and p > 0:
                тоны.append((float(p), cid))
        except Exception:  # noqa
            pass
    тоны.sort()
    fmin, cid = тоны[0]
    период_мс = 1000.0 / fmin
    dt2 = 2 * период_мс
    N = dt2 / 1000 * SR
    N2 = 1 << int(np.ceil(np.log2(N)))
    print(f"тональных клеток: {len(тоны)}")
    print(f"самый низкий тон: {fmin:.1f} Гц ({cid})")
    print(f"1 период = {период_мс:.1f} мс")
    print(f"2 периода = {dt2:.1f} мс = {N:.0f} сэмплов")
    print(f"→ окно кирпича = {N2} сэмплов = {1000*N2/SR:.1f} мс "
          f"(вмещает {N2/SR*fmin:.1f} периода 62 Гц)")


if __name__ == "__main__":
    main()

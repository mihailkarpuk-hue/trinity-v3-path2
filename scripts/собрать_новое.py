# -*- coding: utf-8 -*-
"""Собрать НОВЫЕ звуки (организмы) из библиотеки кирпичей-моментов.

По ПРАВИЛО_СБОРКИ.md / ядро/организм.py.
Использование:
  python3 scripts/собрать_новое.py              # все рецепты
  python3 scripts/собрать_новое.py шторм вьюга  # выбранные
Выход: выход/новое/<имя>.wav
"""
from __future__ import annotations

import hashlib
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ядро.кирпич_момент import SR  # noqa
from ядро.организм import (  # noqa
    РЕЦЕПТЫ,
    загрузить_манифест,
    собрать_по_имени,
    собрать_организм,
    СлойОргана,
)

# совместимость со старым cp_моментов / импортами
_манифест = загрузить_манифест


def кирпичи(cid, органы, сдвиг_с=0.0, лимит=None):
    from ядро.организм import атомы_органа
    return атомы_органа(cid, органы, сдвиг_с=сдвиг_с, лимит=лимит)


def шторм(man):
    return собрать_организм(РЕЦЕПТЫ["шторм"], органы=man)


ВЫХОД = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "выход", "новое",
)


def _wav(path: str, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> None:
    имена = sys.argv[1:] or sorted(РЕЦЕПТЫ.keys())
    man = загрузить_манифест()
    print(f"рецептов: {len(имена)} → {ВЫХОД}")
    for имя in имена:
        s = собрать_по_имени(имя, органы=man)
        путь = os.path.join(ВЫХОД, f"{имя}.wav")
        _wav(путь, s)
        h = hashlib.sha256(s.tobytes()).hexdigest()[:16]
        слои = РЕЦЕПТЫ[имя]
        состав = " + ".join(
            f"{сл.источник}@{сл.сдвиг_с}с×{сл.громкость}"
            + (f"(≤{сл.лимит_атомов})" if сл.лимит_атомов else "")
            for сл in слои
        )
        print(f"  {имя:12s} {len(s)/SR:.2f}с sha={h}  ← {состав}")
        print(f"               → {путь}")
    print("только из кирпичей библиотеки · детерминировано · ПРАВИЛО_СБОРКИ.md")


if __name__ == "__main__":
    main()

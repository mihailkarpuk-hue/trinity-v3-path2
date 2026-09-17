# -*- coding: utf-8
"""Устарело: синтетические формы заменены сборкой из кирпичей по атомам.

Запускайте: python3 scripts/сборка_из_кирpичей.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = next(Path(__file__).parent.glob("сборка_из_кирpичей.py"))
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")

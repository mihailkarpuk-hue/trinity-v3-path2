# -*- coding: utf-8
"""CP#3: сборка звука из кирпичей по скелетам атомов клеток.

Без синтетических форм (спираль/решётка/случайное): только атом-треки
клетки → тип кирпича → ГЕН → wav.

usage:
  python3 scripts/сборка_из_кирпичей.py
  python3 scripts/сборка_из_кирпичей.py etalon_dozhd bazis_12_kaplya
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from ядро.сборка import SR, sha256_wav  # noqa: E402
from ядро.синтез import синтезировать  # noqa: E402
from ядро.словарь_кирпичей import (  # noqa: E402
    КОРПУС_18_БАЗИС,
    загрузить,
    собрать_гены_клетки,
    _cell,
)

ВЫХОД = os.path.join(КОРЕНЬ, "выход", "сборка")
ОТЧЁТ = os.path.join(КОРЕНЬ, "отчёты", "сборка_из_кирпичей.md")

# демо для субъективной проверки + контроль различимости
ДЕМО = (
    "bazis_12_kaplya",
    "etalon_dozhd",
    "etalon_komar",
    "bazis_07_am_etalon",
    "bazis_02_pila_etalon",
    "живая_А",
)


def _write_wav(path: str, y: np.ndarray, sr: int = SR) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())



def _run(cid: str, словарь: dict, *, живость: float = 0.0) -> dict:
    gens, dur, состав = собрать_гены_клетки(cid, словарь, живость=живость)
    y = синтезировать(gens, sr=SR, dur=dur)
    path = os.path.join(ВЫХОД, f"{cid}_кирпичи.wav")
    _write_wav(path, y)
    h1 = sha256_wav(y)
    y2 = синтезировать(gens, sr=SR, dur=dur, meta=meta)
    h2 = sha256_wav(y2)
    return {
        "id": cid,
        "атомов": len(gens),
        "типов": len(состав),
        "состав": состав,
        "wav": path,
        "sha256": h1,
        "детерминизм": h1 == h2,
    }


def main() -> None:
    if "--18" in sys.argv:
        ids = list(КОРПУС_18_БАЗИС)
    else:
        ids = [a for a in sys.argv[1:] if not a.startswith("-")] or list(ДЕМО)
    словарь = загрузить()
    os.makedirs(ВЫХОД, exist_ok=True)
    results = [_run(cid, словарь) for cid in ids]
    hashes = {r["sha256"] for r in results}
    distinct = len(hashes) == len(results)
    det_ok = all(r["детерминизм"] for r in results)
    multi = sum(1 for r in results if r["типов"] >= 2)

    lines = [
        "# Сборка из кирпичей (Этап 5 · CP#3)",
        "",
        f"Дата: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Канон: **скелет атома** (birth, freq, amp, freq_t, amp_t) + **тип кирпича** из словаря.",
        "Синтетические формы (спираль/решётка) **не используются**.",
        "",
        "| клетка | атомов | типов кирпичей | детерминизм | sha256 (16) |",
        "|--------|--------|----------------|-------------|-------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['атомов']} | {r['типов']} | "
            f"{'✓' if r['детерминизм'] else '✗'} | {r['sha256'][:16]}… |"
        )
    lines += ["", "## Состав (кирпичи)", ""]
    for r in results:
        parts = ", ".join(f"{k} {v*100:.0f}%" for k, v in list(r["состав"].items())[:5])
        lines.append(f"- **{r['id']}**: {parts or '—'}")
    cp3 = distinct and det_ok
    lines += [
        "",
        f"- Все sha256 различны: **{'PASS' if distinct else 'FAIL'}** ({len(hashes)}/{len(results)})",
        f"- Детерминизм всех: **{'PASS' if det_ok else 'FAIL'}**",
        f"- Клеток с ≥2 типами кирпичей: {multi}/{len(results)}",
        "",
        f"WAV: `{ВЫХОД}/{{клетка}}_кирпичи.wav`",
        "",
        f"**CP#3:** {'PASS' if cp3 else 'FAIL'}",
    ]
    os.makedirs(os.path.dirname(ОТЧЁТ), exist_ok=True)
    with open(ОТЧЁТ, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for r in results:
        print(f"✓ {r['id']}: {r['типов']} типов · {r['sha256'][:16]}…")
    print(f"→ {ОТЧЁТ}")


if __name__ == "__main__":
    main()

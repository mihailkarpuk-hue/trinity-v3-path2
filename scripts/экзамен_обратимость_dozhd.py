# -*- coding: utf-8 -*-
"""Экзамен обратимости — пилот дождь (путь-2, кусок №6).

Клетка etalon_dozhd (atoms+crosses) → синтез → сравнение с dozhd_real.wav.
Метрики: band_corr (шум/время), stft_corr, FD/nestedness.

Порог смысла: band_corr >= 0.5 → PASS измерения «структура несёт»;
иначе exit 2 (FAIL порога), отчёт всё равно пишется.

Запуск: python3 scripts/экзамен_обратимость_dozhd.py
"""
from __future__ import annotations

import json
import os
import sys
import wave
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    КОРЕНЬ,
    os.path.join(КОРЕНЬ, "ядро"),
    os.path.join(КОРЕНЬ, "экзамен"),
]

from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402
from оси import оси_звука  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402

WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "dozhd_real.wav")
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")
ОТЧЁТЫ = os.path.join(КОРЕНЬ, "отчёты")
OUT_WAV = os.path.join(КОРЕНЬ, "выход", "раскадровка_dozhd_A", "dozhd_synth_A.wav")
ПОРОГ = 0.5


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
        x = x / 32768.0
    return x, sr


def _write_wav(path: str, x: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.clip(x, -1, 1)
    pcm = (y * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main() -> int:
    x, sr = _load_wav(WAV)
    m = float(np.max(np.abs(x)) or 1.0)
    x = x / m

    rec = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms = rec.get("atoms") or []
    crosses = rec.get("crosses")
    dur = len(x) / float(sr)
    y = синтез_из_атомов(atoms, crosses, sr=sr, dur=dur + 0.05, meta=rec)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    ym = float(np.max(np.abs(y)) or 1.0)
    y = y / ym

    band = float(corr(band_spectrogram(x, sr), band_spectrogram(y, sr)))
    stft = float(corr(stft_mag(x, sr), stft_mag(y, sr)))
    o0 = оси_звука(x, sr)
    o1 = оси_звука(y, sr)

    ok_порог = band >= ПОРОГ
    report = {
        "пилот": "dozhd",
        "дата": date.today().isoformat(),
        "mode": "A_sound_cell",
        "alignment_note": "образ_кусочек не в петле этого экзамена",
        "звук_эталон": os.path.relpath(WAV, КОРЕНЬ),
        "клетка": os.path.relpath(КЛЕТКА, КОРЕНЬ),
        "atoms": len(atoms),
        "crosses": len(crosses or []),
        "sr": sr,
        "sec": round(n / sr, 4),
        "band_corr": round(band, 4),
        "stft_corr": round(stft, 4),
        "fd_orig": o0.get("fd"),
        "fd_synth": o1.get("fd"),
        "nestedness_orig": o0.get("nestedness"),
        "nestedness_synth": o1.get("nestedness"),
        "порог_band_corr": ПОРОГ,
        "вердикт_порога": "PASS" if ok_порог else "FAIL",
        "synth_wav": os.path.relpath(OUT_WAV, КОРЕНЬ),
    }

    _write_wav(OUT_WAV, y, sr)
    os.makedirs(ОТЧЁТЫ, exist_ok=True)
    jp = os.path.join(ОТЧЁТЫ, "экзамен_обратимость_dozhd.json")
    mp = os.path.join(ОТЧЁТЫ, "экзамен_обратимость_dozhd.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Экзамен обратимости — дождь (звук↔атомы клетки)",
        "",
        f"> {report['дата']} · порог band_corr ≥ {ПОРОГ}",
        "",
        f"**Вердикт порога:** **{report['вердикт_порога']}**",
        "",
        "| метрика | значение |",
        "|---------|----------|",
        f"| band_corr (врем-спектрограмма) | **{report['band_corr']}** |",
        f"| stft_corr | {report['stft_corr']} |",
        f"| FD orig → synth | {report['fd_orig']} → {report['fd_synth']} |",
        f"| nestedness orig → synth | {report['nestedness_orig']} → {report['nestedness_synth']} |",
        f"| атомов / крестов | {report['atoms']} / {report['crosses']} |",
        "",
        f"Синтез: `{report['synth_wav']}`",
        "",
        "**Честно:** петля без образ_кусочек clean; полный синхрон-атом — следующий контур.",
        f"Справка канона букв: FD round-trip r≈0.972 — здесь шум дождя, другая шкала.",
        "",
    ]
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(report, ensure_ascii=False))
    return 0 if ok_порог else 2


if __name__ == "__main__":
    raise SystemExit(main())

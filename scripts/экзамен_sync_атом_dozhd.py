# -*- coding: utf-8 -*-
"""Экзамен sync-атома дождь A (кусок B№4).

1) Все образ_кусочек.path существуют
2) band_corr звук клетки → synth vs dozhd_real.wav (порог 0.5)
3) Отчёт: 12 атомов + честный флаг образ_в_синтезе=false

Запуск: python3 scripts/экзамен_sync_атом_dozhd.py
"""
from __future__ import annotations

import json
import os
import sys
import wave
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро"), os.path.join(КОРЕНЬ, "экзамен")]

from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402
from оси import оси_звука  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402

PACKET = os.path.join(КОРЕНЬ, "выход", "sync_atoms_dozhd_A.json")
WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "dozhd_real.wav")
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")
ОТЧЁТЫ = os.path.join(КОРЕНЬ, "отчёты")
ПОРОГ = 0.5


def _load_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    return x, sr


def main() -> int:
    packet = json.load(open(PACKET, encoding="utf-8"))
    atoms_sync = packet.get("atoms") or []
    missing = []
    rows = []
    for a in atoms_sync:
        p = (a.get("образ_кусочек") or {}).get("path")
        full = os.path.join(КОРЕНЬ, p) if p else ""
        ok = bool(p) and os.path.isfile(full)
        if not ok:
            missing.append(a.get("id"))
        rows.append(
            {
                "id": a.get("id"),
                "birth": a.get("birth"),
                "frame_i": a.get("frame_i"),
                "path": p,
                "path_ok": ok,
                "долг_params_104": (a.get("звук_ядро") or {}).get("долг_params_104"),
            }
        )

    x, sr = _load_wav(WAV)
    m = float(np.max(np.abs(x)) or 1.0)
    x = x / m
    rec = json.load(open(КЛЕТКА, encoding="utf-8"))
    y = синтез_из_атомов(rec.get("atoms"), rec.get("crosses"), sr=sr, dur=len(x) / sr + 0.05, meta=rec)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    y = y / (float(np.max(np.abs(y)) or 1.0))
    band = float(corr(band_spectrogram(x, sr), band_spectrogram(y, sr)))
    stft = float(corr(stft_mag(x, sr), stft_mag(y, sr)))
    o0, o1 = оси_звука(x, sr), оси_звука(y, sr)

    crops_ok = len(missing) == 0 and len(atoms_sync) >= 12
    sound_ok = band >= ПОРОГ
    verdict = "PASS" if crops_ok and sound_ok else "FAIL"

    report = {
        "дата": date.today().isoformat(),
        "вердикт": verdict,
        "образ_в_синтезе": False,
        "образ_привязан_файлами": crops_ok,
        "n_sync_atoms": len(atoms_sync),
        "crops_missing": missing,
        "band_corr": round(band, 4),
        "stft_corr": round(stft, 4),
        "fd_orig": o0.get("fd"),
        "fd_synth": o1.get("fd"),
        "порог_band_corr": ПОРОГ,
        "atoms": rows,
        "note": "звук: петля клетки; образ: наличие crops, не вход синтеза",
    }

    os.makedirs(ОТЧЁТЫ, exist_ok=True)
    jp = os.path.join(ОТЧЁТЫ, "экзамен_sync_атом_dozhd.json")
    mp = os.path.join(ОТЧЁТЫ, "экзамен_sync_атом_dozhd.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Экзамен sync-атом — дождь A",
        "",
        f"> {report['дата']} · **{verdict}**",
        "",
        f"- образ_в_синтезе: **false** (честно)",
        f"- crops на диске: **{crops_ok}** (N={len(atoms_sync)}, missing={missing or '—'})",
        f"- band_corr звук клетки: **{report['band_corr']}** (порог {ПОРОГ})",
        f"- stft_corr: {report['stft_corr']} · FD {report['fd_orig']}→{report['fd_synth']}",
        "",
        "| id | birth | frame | path_ok |",
        "|----|------:|------:|:-------:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['id']}` | {r['birth']} | {r['frame_i']} | {r['path_ok']} |"
        )
    lines += [
        "",
        "## Для автора (кусок E)",
        "",
        "- Послушать: `выход/раскадровка_dozhd_A/dozhd_synth_A.wav` (или пересобрать тем же синтезом)",
        "- Посмотреть: `выход/sync_atoms_dozhd_A/crops/`",
        "- Ответить: ок / не ок / что править",
        "",
    ]
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"вердикт": verdict, "band_corr": report["band_corr"], "crops_ok": crops_ok}, ensure_ascii=False))
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

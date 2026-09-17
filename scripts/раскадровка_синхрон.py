# -*- coding: utf-8 -*-
"""Раскадровка синхрон режим A — пилот дождь.

birth (клетка etalon_dozhd) → frame_i = floor(birth * fps) на clean_video.
Контракт: отчёты/контракт_синхрон_атом.md
alignment = assumed_t0

Запуск: python3 scripts/раскадровка_синхрон.py
"""
from __future__ import annotations

import json
import os
import wave
from datetime import date

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ОТЧЁТЫ = os.path.join(КОРЕНЬ, "отчёты")
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")
WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "dozhd_real.wav")
CLEAN = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "clean_video.mp4")
CLEAN_META = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "clean_meta.json")


def _wav_sec(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _video_meta(path: str, meta_path: str) -> tuple[float, int]:
    fps, frames = None, None
    if os.path.isfile(meta_path):
        m = json.load(open(meta_path, encoding="utf-8"))
        fps = float(m.get("fps") or 0) or None
        frames = int(m.get("кадров") or 0) or None
    if fps is None or frames is None:
        import cv2

        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    return fps, frames


def main() -> int:
    wav_sec = _wav_sec(WAV)
    fps, n_frames = _video_meta(CLEAN, CLEAN_META)
    if not fps or not n_frames:
        raise SystemExit("нет fps/frames у clean_video")

    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms = cell.get("atoms") or []
    births = sorted({round(float(a.get("birth") or 0), 6) for a in atoms})

    rows = []
    rejected = 0
    for birth in births:
        if birth < 0 or birth >= wav_sec:
            rejected += 1
            continue
        frame_i = int(birth * fps)
        if frame_i < 0 or frame_i >= n_frames:
            rejected += 1
            continue
        rows.append(
            {
                "birth": birth,
                "frame_i": frame_i,
                "t_frame": round(frame_i / fps, 6),
            }
        )

    report = {
        "пилот": "dozhd",
        "дата": date.today().isoformat(),
        "mode": "A",
        "alignment": "assumed_t0",
        "звук": os.path.relpath(WAV, КОРЕНЬ),
        "clean_video": os.path.relpath(CLEAN, КОРЕНЬ),
        "wav_sec": round(wav_sec, 4),
        "fps": fps,
        "frames_clean": n_frames,
        "unique_births_in_cell": len(births),
        "rows_ok": len(rows),
        "rows_rejected": rejected,
        "atoms_in_cell": len(atoms),
        "rows": rows,
    }

    os.makedirs(ОТЧЁТЫ, exist_ok=True)
    jp = os.path.join(ОТЧЁТЫ, "раскадровка_dozhd_A.json")
    mp = os.path.join(ОТЧЁТЫ, "раскадровка_dozhd_A.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    sample = rows[:8]
    lines = [
        "# Раскадровка дождь — режим A",
        "",
        f"> {report['дата']} · mode=A · alignment=assumed_t0",
        "",
        f"- wav: `{report['звук']}` ({report['wav_sec']} с)",
        f"- clean: `{report['clean_video']}` · fps={fps} · кадров={n_frames}",
        f"- уникальных birth в клетке: {len(births)}",
        f"- строк OK: **{len(rows)}** · отброшено: {rejected}",
        f"- атомов в клетке: {len(atoms)}",
        "",
        "## Образец строк",
        "",
        "| birth | frame_i | t_frame |",
        "|------:|--------:|--------:|",
    ]
    for r in sample:
        lines.append(f"| {r['birth']} | {r['frame_i']} | {r['t_frame']} |")
    if len(rows) > len(sample):
        lines.append(f"| … | … | ещё {len(rows) - len(sample)} |")
    lines += ["", f"JSON: `{os.path.relpath(jp, КОРЕНЬ)}`", ""]
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"ok": True, "rows_ok": len(rows), "json": jp}, ensure_ascii=False))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

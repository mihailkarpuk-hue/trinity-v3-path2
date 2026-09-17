# -*- coding: utf-8 -*-
"""Инвентарь осей синхрона — пилот дождь (путь-2).

Только факты: длительности звука / видео / span birth атомов + флаги mismatch.
Без раскадровки и без сети.

Запуск из Тринити_V3:
  python3 scripts/инвентарь_синхрон_пилот.py
"""
from __future__ import annotations

import json
import os
import subprocess
import wave
from datetime import date

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
ОТЧЁТЫ = os.path.join(КОРЕНЬ, "отчёты")
СТИХ = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd")
ЭТАЛОН = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны")
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")


def _wav_sec(path: str) -> dict:
    with wave.open(path, "rb") as w:
        n, rate = w.getnframes(), w.getframerate()
        return {
            "path": os.path.relpath(path, КОРЕНЬ),
            "sec": round(n / rate, 4) if rate else None,
            "sr": rate,
            "channels": w.getnchannels(),
            "exists": True,
        }


def _ffprobe(path: str) -> dict:
    out = {
        "path": os.path.relpath(path, КОРЕНЬ),
        "exists": os.path.isfile(path),
        "sec": None,
        "fps": None,
        "frames": None,
        "has_audio": False,
        "audio_sec": None,
    }
    if not out["exists"]:
        return out
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        out["fps"] = round(fps, 4) if fps else None
        out["frames"] = int(frames) if frames else None
        out["sec"] = round(frames / fps, 4) if fps and frames else None
    except Exception as e:
        out["cv2_error"] = str(e)

    ff = FFMPEG if os.path.isfile(FFMPEG) else "ffprobe"
    try:
        cmd = [
            ff if ff == "ffprobe" else FFMPEG,
            "-hide_banner",
            "-i",
            path,
        ]
        # ffmpeg prints probe on stderr
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        blob = (r.stderr or "") + (r.stdout or "")
        out["has_audio"] = "Audio:" in blob
        for line in blob.splitlines():
            if "Duration:" in line:
                # Duration: 00:00:09.94,
                part = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = part.split(":")
                out["container_sec"] = round(
                    int(h) * 3600 + int(m) * 60 + float(s), 4
                )
                break
        if out["has_audio"] and out.get("container_sec") is not None:
            out["audio_sec"] = out["container_sec"]
    except Exception as e:
        out["ffmpeg_error"] = str(e)
    return out


def _atoms_span(path: str) -> dict:
    info = {"path": os.path.relpath(path, КОРЕНЬ), "exists": os.path.isfile(path)}
    if not info["exists"]:
        return info
    data = json.load(open(path, encoding="utf-8"))
    atoms = data.get("atoms") or []
    births = [float(a.get("birth") or 0) for a in atoms]
    info.update(
        {
            "atoms_count": len(atoms),
            "birth_min": round(min(births), 6) if births else None,
            "birth_max": round(max(births), 6) if births else None,
            "span_sec": round(max(births) - min(births), 6) if births else None,
            "длительность_сек_meta": data.get("длительность_сек"),
        }
    )
    return info


def main() -> int:
    wav = _wav_sec(os.path.join(ЭТАЛОН, "dozhd_real.wav"))
    videos = {
        name: _ffprobe(os.path.join(СТИХ, name))
        for name in ("video_live_01.mp4", "video_live_02.mp4", "clean_video.mp4")
    }
    model3d = _atoms_span(os.path.join(ЭТАЛОН, "dozhd_model3d.json"))
    cell = _atoms_span(КЛЕТКА)

    live = videos.get("video_live_02.mp4") or {}
    live_sec = live.get("sec") or live.get("container_sec")
    wav_sec = wav.get("sec")
    atom_span = model3d.get("span_sec")

    mismatches = {
        "mismatch_wav_vs_video_live_02": bool(
            wav_sec is not None
            and live_sec is not None
            and abs(wav_sec - live_sec) > 0.25
        ),
        "mismatch_wav_vs_atom_span_model3d": bool(
            wav_sec is not None
            and atom_span is not None
            and abs(wav_sec - atom_span) > 0.25
        ),
        "mismatch_video_vs_atom_span_model3d": bool(
            live_sec is not None
            and atom_span is not None
            and abs(live_sec - atom_span) > 0.25
        ),
        "note": (
            "Эталон dozhd_real.wav короче video_live_02 — "
            "нельзя считать их одной осью без явного правила."
            if wav_sec and live_sec and abs(wav_sec - (live_sec or 0)) > 0.25
            else "длительности близки или данных нет"
        ),
    }

    report = {
        "пилот": "dozhd",
        "дата": date.today().isoformat(),
        "путь": "путь-2-момент",
        "звук_эталон": wav,
        "видео": videos,
        "атомы_model3d": model3d,
        "атомы_клетка": cell,
        "mismatches": mismatches,
    }

    os.makedirs(ОТЧЁТЫ, exist_ok=True)
    json_path = os.path.join(ОТЧЁТЫ, "синхрон_пилот_dozhd.json")
    md_path = os.path.join(ОТЧЁТЫ, "синхрон_пилот_dozhd.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def row(label, sec, extra=""):
        return f"| {label} | {sec if sec is not None else '—'} | {extra} |"

    lines = [
        "# Синхрон-пилот дождь — инвентарь осей",
        "",
        f"> Дата: {report['дата']} · путь-2 · только факты",
        "",
        "| Ось | сек | заметка |",
        "|-----|-----|---------|",
        row("dozhd_real.wav", wav_sec, f"sr={wav.get('sr')}"),
        row(
            "video_live_02",
            live_sec,
            f"fps={live.get('fps')} audio={live.get('has_audio')}",
        ),
        row(
            "clean_video",
            (videos.get("clean_video.mp4") or {}).get("sec"),
            f"fps={(videos.get('clean_video.mp4') or {}).get('fps')}",
        ),
        row(
            "atoms model3d span",
            atom_span,
            f"n={model3d.get('atoms_count')} max_birth={model3d.get('birth_max')}",
        ),
        row(
            "atoms клетка span",
            cell.get("span_sec"),
            f"n={cell.get('atoms_count')} max_birth={cell.get('birth_max')}",
        ),
        "",
        "## Флаги рассинхрона",
        "",
        f"- `mismatch_wav_vs_video_live_02`: **{mismatches['mismatch_wav_vs_video_live_02']}**",
        f"- `mismatch_wav_vs_atom_span_model3d`: **{mismatches['mismatch_wav_vs_atom_span_model3d']}**",
        f"- `mismatch_video_vs_atom_span_model3d`: **{mismatches['mismatch_video_vs_atom_span_model3d']}**",
        "",
        f"**Заметка:** {mismatches['note']}",
        "",
        f"JSON: `{os.path.relpath(json_path, КОРЕНЬ)}`",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"ok": True, "md": md_path, "json": json_path, "mismatches": mismatches}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

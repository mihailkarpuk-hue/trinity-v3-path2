# -*- coding: utf-8 -*-
"""Инвентарь видимости причины звука дождя (капля→удар). Кусок карты причины №1.

Пишет отчёты/причина_дождь_инвентарь.md и обновляет probe-кадры.
Без детектора событий.

Запуск: python3 scripts/инвентарь_причина_дождь.py
"""
from __future__ import annotations

import json
import os
from datetime import date

import cv2

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOZHD = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd")
ОТЧЁТ = os.path.join(КОРЕНЬ, "отчёты", "причина_дождь_инвентарь.md")

# Эвристика «глаз скрипта» + жёсткие правила автора (зафиксированы после просмотра):
# video_live_01 — лужа/асфальт: видны полосы капель, круги, всплески → ПРИГОДЕН
# video_live_02 — кроны деревьев: капля/удар не читаются → НЕПРИГОДЕН как причина
# clean из live_02 — маска листвы, не капля→удар → НЕПРИГОДЕН как причина

VERDICT = {
    "video_live_01.mp4": {
        "пригоден": True,
        "почему": "поверхность с водой: полосы летящих капель, круги ряби, короны всплеска — видна причина звука (капля→удар)",
        "событие": "drop_impact",
    },
    "video_live_02.mp4": {
        "пригоден": False,
        "почему": "кроны/лес: атмосфера дождя есть, отдельная капля и удар о поверхность не читаются",
        "событие": None,
    },
    "clean_video.mp4": {
        "пригоден": False,
        "почему": "clean из live_02 — маска листвы/зерно, не физическая капля→удар (ошибка пилота B)",
        "событие": None,
        "источник_clean": "video_live_02.mp4",
    },
}


def _probe(name: str) -> dict:
    path = os.path.join(DOZHD, name)
    info = {"файл": name, "exists": os.path.isfile(path)}
    if not info["exists"]:
        return info
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    mid = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    probe = os.path.join(DOZHD, f"_probe_причина_{name}.jpg")
    if ok and frame is not None:
        cv2.imwrite(probe, frame)
        info["probe"] = os.path.relpath(probe, КОРЕНЬ)
    info.update({"w": w, "h": h, "fps": fps, "frames": n, "sec": round(n / fps, 3) if fps else None})
    v = VERDICT.get(name, {})
    info.update(v)
    return info


def main() -> int:
    rows = [_probe(n) for n in ("video_live_01.mp4", "video_live_02.mp4", "clean_video.mp4")]
    rec = next((r for r in rows if r.get("пригоден")), None)

    lines = [
        "# Инвентарь: где видна причина звука дождя (капля→удар)",
        "",
        f"> {date.today().isoformat()} · после E: образ ≠ 2D-маска, а **частичка мироздания**",
        "",
        "## Правило автора",
        "",
        "Звук дождя создаёт **летящая капля**, которая **ударяется о поверхность**.",
        "Образ_кусочек должен показывать эту причину, не произвольный кадр.",
        "",
        "## Клипы",
        "",
        "| файл | пригоден | сек | почему |",
        "|------|:--------:|----:|--------|",
    ]
    for r in rows:
        if not r.get("exists"):
            lines.append(f"| `{r['файл']}` | — | — | нет файла |")
            continue
        flag = "**да**" if r.get("пригоден") else "**нет**"
        lines.append(
            f"| `{r['файл']}` | {flag} | {r.get('sec')} | {r.get('почему')} |"
        )

    lines += [
        "",
        "## Рекомендация для следующих кусков",
        "",
    ]
    if rec:
        lines += [
            f"**Источник причины:** `{rec['файл']}`",
            f"- событие: `{rec.get('событие')}`",
            f"- probe: `{rec.get('probe')}`",
            "",
            "**Не использовать как причину:** `video_live_02.mp4`, `clean_video.mp4` (пилот B).",
            "",
            "Звук: для связи с ударами предпочтителен **родной AAC** этого клипа (режим ближе к B),",
            "а не короткий `dozhd_real.wav` с другой оси — иначе снова assumed_t0 между разными мирами.",
            "",
        ]
    else:
        lines += ["**Нет пригодного клипа** — нужен новый медиа-поиск (отдельный GATE).", ""]

    lines += [
        "## Следствие",
        "",
        "Карта B (crops с clean live_02) закрыта как мимо смысла.",
        "Дальше: контракт события + детектор на **live_01**.",
        "",
    ]
    os.makedirs(os.path.dirname(ОТЧЁТ), exist_ok=True)
    with open(ОТЧЁТ, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    summary = {
        "рекомендация": rec["файл"] if rec else None,
        "пригодные": [r["файл"] for r in rows if r.get("пригоден")],
        "непригодные": [r["файл"] for r in rows if r.get("exists") and not r.get("пригоден")],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if rec else 2


if __name__ == "__main__":
    raise SystemExit(main())

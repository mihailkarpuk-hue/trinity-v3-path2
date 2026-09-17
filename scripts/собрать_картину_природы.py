# -*- coding: utf-8 -*-
"""Собрать НОВУЮ картину из кусочков библиотеки_природы.

Буква ≠ картина. Здесь проверка: атомы разных стихий → один организм
(звук) + один составной образ (частички кадров).

Рецепт по умолчанию «лесная_гроза»:
  ветер + дождь + гром + вспышка молнии — такого цельного клипа в источниках нет.

Запуск:
  python3 scripts/собрать_картину_природы.py
  python3 scripts/собрать_картину_природы.py лесная_гроза
"""
from __future__ import annotations

import json
import os
import sys
import wave
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)

from ядро.кирпич_момент import ОКНО, SR, из_кирпичей  # noqa: E402
from ядро.организм import СлойОргана, собрать_организм  # noqa: E402

БИБ = os.path.join(КОРЕНЬ, "данные", "библиотека_природы")
ВЫХОД = os.path.join(КОРЕНЬ, "выход", "картины")

# Цвета слоёв для составного образа (не freqToHue — метка источника)
ЦВЕТ_СТИХИИ = {
    "veter": (180, 200, 220),
    "dozhd": (140, 190, 255),
    "grom": (255, 210, 120),
    "molniya": (255, 255, 200),
    "ogon": (255, 140, 40),
    "reka": (80, 160, 200),
    "vodopad": (160, 220, 255),
}

РЕЦЕПТЫ: dict[str, list[СлойОргана]] = {
    # новая картина: не один исходный ролик
    "лесная_гроза": [
        СлойОргана("veter", 0.0, 0.55, лимит_атомов=80),
        СлойОргана("dozhd", 0.8, 0.85, лимит_атомов=120),
        СлойОргана("grom", 2.4, 1.0, лимит_атомов=40),
        СлойОргана("molniya", 2.6, 0.7, лимит_атомов=30),
    ],
    "костёр_у_реки": [
        СлойОргана("reka", 0.0, 0.5, лимит_атомов=90),
        СлойОргана("ogon", 0.6, 1.0, лимит_атомов=100),
        СлойОргана("veter", 1.0, 0.4, лимит_атомов=50),
    ],
}


def загрузить_органы_природы(путь: str = БИБ) -> dict[str, dict]:
    """манифест природы → органы с атомами (как ждёт ядро/организм)."""
    with open(os.path.join(путь, "манифест.json"), encoding="utf-8") as f:
        M = json.load(f)
    out: dict[str, dict] = {}
    for o in M["органы"]:
        cid = o["стихия_id"]
        full = json.load(open(os.path.join(путь, o["файл"]), encoding="utf-8"))
        out[cid] = {
            "источник": cid,
            "атомы": full["атомы"],
            "имя": full.get("имя") or cid,
        }
    return out


def wav_записать(path: str, y: np.ndarray, sr: int = SR) -> None:
    y = np.asarray(y, dtype=np.float64)
    peak = float(np.max(np.abs(y))) + 1e-12
    y16 = (y / peak * 0.9 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y16.tobytes())


def собрать_образ_картины(
    слои: list[СлойОргана],
    органы: dict[str, dict],
    *,
    W: int = 900,
    H: int = 900,
) -> tuple[np.ndarray, list[dict]]:
    """Составной образ: частички атомов слоёв, сдвиг по времени = сдвиг по X."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    meta_pts: list[dict] = []
    # длительность картины для нормировки X
    tmax = 0.0
    for слой in слои:
        атомы = органы[слой.источник]["атомы"]
        if слой.лимит_атомов is not None:
            атомы = атомы[: слой.лимит_атомов]
        for a in атомы:
            tmax = max(tmax, float(a["t"]) + float(слой.сдвиг_с))
    tmax = max(tmax, 1e-6)

    for слой in слои:
        cid = слой.источник
        base = ЦВЕТ_СТИХИИ.get(cid, (200, 200, 200))
        атомы = органы[cid]["атомы"]
        if слой.лимит_атомов is not None:
            атомы = атомы[: слой.лимит_атомов]
        # не все атомы — каждый N-й, чтобы не забить холст
        step = max(1, len(атомы) // 40)
        for a in атомы[::step]:
            t = float(a["t"]) + float(слой.сдвиг_с)
            x_off = t / tmax  # 0..1 вдоль времени картины
            img = a.get("образ_кусочек") or {}
            for p in (img.get("точки") or [])[:4]:
                # локальный x кадра сжимаем и сдвигаем слоем времени
                lx = float(p.get("x") or 0.5) * 0.22
                x = int(round((x_off * 0.78 + lx) * (W - 1)))
                y = int(round(float(p.get("y") or 0.5) * (H - 1)))
                x = max(0, min(W - 1, x))
                y = max(0, min(H - 1, y))
                # смесь цвета стихии и цвета точки кадра
                pr, pg, pb = int(p.get("r", 255)), int(p.get("g", 255)), int(p.get("b", 255))
                r = int(0.45 * base[0] + 0.55 * pr)
                g = int(0.45 * base[1] + 0.55 * pg)
                b = int(0.45 * base[2] + 0.55 * pb)
                rad = max(2, int(3 + 10 * float(p.get("размер") or 0.4) * float(слой.громкость)))
                cv2.circle(canvas, (x, y), rad, (b, g, r), -1, lineType=cv2.LINE_AA)
                meta_pts.append({
                    "стихия": cid, "t": round(t, 4),
                    "x": round(x / (W - 1), 4), "y": round(y / (H - 1), 4),
                    "r": r, "g": g, "b": b,
                })
    return canvas, meta_pts


def main() -> None:
    имя = sys.argv[1] if len(sys.argv) > 1 else "лесная_гроза"
    if имя not in РЕЦЕПТЫ:
        raise SystemExit(f"нет рецепта «{имя}»; есть: {sorted(РЕЦЕПТЫ)}")

    os.makedirs(ВЫХОД, exist_ok=True)
    органы = загрузить_органы_природы(БИБ)
    слои = РЕЦЕПТЫ[имя]

    # звук
    y = собрать_организм(слои, органы=органы, путь_биб=БИБ)
    wav_path = os.path.join(ВЫХОД, f"{имя}.wav")
    wav_записать(wav_path, y)

    # образ
    canvas, pts = собрать_образ_картины(слои, органы)
    png_path = os.path.join(ВЫХОД, f"{имя}.png")
    cv2.imwrite(png_path, canvas)

    meta = {
        "id": имя,
        "дата": str(date.today()),
        "правило": "буква ≠ картина; картина = сборка атомов (тип ≠ место)",
        "состав": [
            {
                "стихия": s.источник,
                "сдвиг_с": s.сдвиг_с,
                "громкость": s.громкость,
                "лимит_атомов": s.лимит_атомов,
            }
            for s in слои
        ],
        "длит_с": round(len(y) / SR, 3),
        "точек_образа": len(pts),
        "wav": f"../../выход/картины/{имя}.wav",
        "png": f"../../выход/картины/{имя}.png",
        "полка": "данные/библиотека_природы",
    }
    with open(os.path.join(ВЫХОД, f"{имя}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"картина «{имя}»: {meta['длит_с']} с, {meta['точек_образа']} точек")
    print(f"  звук  → {wav_path}")
    print(f"  образ → {png_path}")
    for s in слои:
        print(f"  слой  {s.источник} @ {s.сдвиг_с}s ×{s.громкость}")


if __name__ == "__main__":
    main()

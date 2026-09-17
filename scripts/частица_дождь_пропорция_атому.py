# -*- coding: utf-8 -*-
"""Мельчайшая частица мироздания дождя — пропорциональна size звукового атома.

Автор: не пейзаж из видео, а частица масштаба атома клетки.
Видео даёт только момент/геометрию удара; радиус/масса/импульс — из
пропорции к atom.size + физика капли.

Пропорция (пилот):
  size_med клетки ≈ 0.113  ↔  r_ref = 1.0 мм (типичная дождевая)
  r_m = (size / size_med) * r_ref

Запуск: python3 scripts/частица_дождь_пропорция_атому.py
"""
from __future__ import annotations

import json
import math
import os
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")
TRACK = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.json")
OUT_JSON = os.path.join(КОРЕНЬ, "выход", "частица_дождь_атом_масштаб.json")
OUT_WAV = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "частица_атом_масштаб.wav")
OUT_PNG = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "частица_атом_масштаб.png")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "частица_дождь_атом_масштаб.md")

SR = 22050
R_REF_M = 0.001  # 1 мм ↔ median size
RHO = 1000.0
G = 9.81


def _write_wav(path: str, x: np.ndarray, sr: int = SR):
    import wave

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def terminal_v(r_m: float) -> float:
    """Грубая терминальная скорость дождевой капли (м/с), эмпирика по радиусу."""
    # Best ~: d=2r в мм → v ≈ 0.5..9
    d_mm = 2 * r_m * 1000.0
    v = 0.2 + 4.0 * (d_mm ** 0.7)
    return float(min(9.5, max(0.8, v)))


def synth_drop(r_m: float, v_ms: float, amp_atom: float) -> tuple[np.ndarray, dict]:
    m = (4.0 / 3.0) * math.pi * (r_m ** 3) * RHO
    e_kin = 0.5 * m * v_ms ** 2
    # громкость связана и с amp атома, и с энергией
    amp = float(min(0.9, 0.08 + 0.55 * amp_atom + 0.12 * math.log10(e_kin * 1e12 + 1)))
    tau = 0.0004 + 25.0 * r_m
    f0 = 2000.0 * (0.001 / max(r_m, 1e-5)) ** 0.5  # меньше капля → выше щёлчок
    f0 = float(min(6000.0, max(600.0, f0)))
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    env = np.exp(-t / max(tau, 1e-5))
    y = amp * env * np.sin(2 * math.pi * f0 * t)
    y += 0.25 * amp * np.exp(-t / (tau * 3)) * np.sin(2 * math.pi * f0 * 0.4 * t)
    y = y / (np.max(np.abs(y)) + 1e-12) * amp
    meta = {
        "r_m": r_m,
        "r_mm": round(r_m * 1000, 4),
        "v_ms": round(v_ms, 3),
        "m_kg": m,
        "e_kin_j": e_kin,
        "f0_hz": round(f0, 1),
        "tau_s": round(tau, 6),
        "amp": round(amp, 4),
    }
    return y.astype(np.float64), meta


def render_particle(r_m: float, size_atom: float, path: str):
    """Крошечный кадр: только сфера-капля, масштаб ~ size атома (не пейзаж)."""
    # canvas пропорционален size: базовый 64px при size=0.113
    side = int(max(24, min(96, 64 * (size_atom / 0.113))))
    img = np.zeros((side, side, 3), dtype=np.uint8)
    cx = cy = side // 2
    # радиус круга в пикселях пропорционален r / r_ref
    rad = max(2, int((side * 0.35) * (r_m / R_REF_M)))
    rad = min(rad, side // 2 - 1)
    # капля: градиент «вода»
    for y in range(side):
        for x in range(side):
            d = math.hypot(x - cx, y - cy)
            if d <= rad:
                # блик сверху
                shade = 180 + int(60 * (1 - d / rad))
                highlight = 40 if (y < cy - rad * 0.3 and abs(x - cx) < rad * 0.4) else 0
                g = min(255, shade + highlight)
                img[y, x] = (g - 40, g - 10, g)  # BGR стекло-вода
    cv2.circle(img, (cx, cy), rad, (220, 220, 220), 1)
    cv2.putText(img, f"{r_m*1000:.2f}mm", (2, side - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def main() -> int:
    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms = cell["atoms"]
    sizes = [float(a.get("size") or 0.11) for a in atoms]
    size_med = float(np.median(sizes))

    track = json.load(open(TRACK, encoding="utf-8")) if os.path.isfile(TRACK) else {}
    t_imp = float(track.get("t_impact") or 2.2)

    # атом клетки того же масштаба времени (ближайший birth, медианный size в окне)
    window = [a for a in atoms if abs(float(a.get("birth") or 0) - t_imp) <= 0.03]
    if not window:
        window = atoms
    # выбрать атом с size близким к медиане окна (типичная «мельчайшая» единица)
    sizes_w = [float(a.get("size") or size_med) for a in window]
    target = float(np.median(sizes_w))
    ref_atom = min(window, key=lambda a: abs(float(a.get("size") or 0) - target))

    size_a = float(ref_atom.get("size") or size_med)
    amp_a = float(ref_atom.get("amp") or 0.1)
    # пропорция
    r_m = (size_a / size_med) * R_REF_M
    r_m = float(max(2e-4, min(0.0025, r_m)))  # 0.2–2.5 мм
    v_ms = terminal_v(r_m)

    y, phys = synth_drop(r_m, v_ms, amp_a)
    _write_wav(OUT_WAV, y, SR)
    render_particle(r_m, size_a, OUT_PNG)

    particle = {
        "id": "particle_dozhd_atom_scale_001",
        "тип": "мельчайшая_частица_мироздания",
        "стихия": "дождь",
        "дата": date.today().isoformat(),
        "пропорция": {
            "формула": "r_m = (atom.size / size_med) * 1mm",
            "size_med_клетки": round(size_med, 4),
            "atom_size": size_a,
            "r_ref_m": R_REF_M,
            "смысл": "частица мира сопоставима размеру звукового атома пропорционально",
        },
        "звук_атом_референс": {
            "клетка": "etalon_dozhd",
            "birth": ref_atom.get("birth"),
            "size": size_a,
            "amp": amp_a,
            "freq": ref_atom.get("freq"),
            "pos": [ref_atom.get("pos_x"), ref_atom.get("pos_y"), ref_atom.get("pos_z")],
        },
        "геометрия_видео": {
            "t_impact": t_imp,
            "xy_impact": track.get("xy_impact"),
            "роль": "только момент/место удара, не размер пейзажа",
        },
        "физика": phys,
        "файлы": {
            "wav": os.path.relpath(OUT_WAV, КОРЕНЬ),
            "png_только_капля": os.path.relpath(OUT_PNG, КОРЕНЬ),
        },
        "не_есть": "кроп листвы/облака/лужи целиком",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(particle, f, ensure_ascii=False, indent=2)

    lines = [
        "# Мельчайшая частица дождя ∝ size атома",
        "",
        f"> size_atom={size_a} / med={size_med:.4f} → **r={phys['r_mm']} мм** · v={phys['v_ms']} м/с · f0={phys['f0_hz']} Гц",
        "",
        f"- json: `{os.path.relpath(OUT_JSON, КОРЕНЬ)}`",
        f"- wav: `{particle['файлы']['wav']}`",
        f"- png (только капля): `{particle['файлы']['png_только_капля']}`",
        f"- референс-атом birth={ref_atom.get('birth')} freq={ref_atom.get('freq')}",
        "",
        "Слушай wav и смотри png — это масштаб атома, не кадр пейзажа.",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({"ok": True, "r_mm": phys["r_mm"], "size": size_a, "wav": OUT_WAV}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""5 секунд звука с video_live_01 → сборка из частиц ∝ атомам.

1) Вырезать 5с native AAC
2) События-капли в [0,5) с детектора / добор
3) На каждое событие — частица масштаба атома (size→r) + импульс по физике
4) Плюс частицы из birth атомов клетки (0..2.4с) + тихая фактура синтезом
5) Свести микс → сравнить с native (band_corr)

Запуск: python3 scripts/сборка_5с_из_частиц_атомов.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from datetime import date
from sys import path as sys_path

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")
EVENTS = os.path.join(КОРЕНЬ, "отчёты", "события_drop_impact_dozhd.json")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")

OUT_DIR = os.path.join(КОРЕНЬ, "выход", "сборка_5с_частицы")
OUT_NATIVE = os.path.join(OUT_DIR, "native_5s.wav")
OUT_MIX = os.path.join(OUT_DIR, "сборка_из_частиц_атомов.wav")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "сборка_5с_частицы_атомы.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "сборка_5с_частицы_атомы.md")

SR = 22050
DUR = 5.0
T0 = 0.0
R_REF = 0.001
RHO = 1000.0


def _ff():
    return FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"


def extract_native(t0: float, dur: float, path: str) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = [
        _ff(), "-y", "-ss", f"{t0:.3f}", "-i", VIDEO,
        "-t", f"{dur:.3f}", "-ac", "1", "-ar", str(SR), path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def read_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    return x, sr


def write_wav(path: str, x: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def terminal_v(r_m: float) -> float:
    d_mm = 2 * r_m * 1000.0
    return float(min(9.5, max(0.8, 0.2 + 4.0 * (d_mm ** 0.7))))


def impulse(r_m: float, v_ms: float, amp_atom: float) -> np.ndarray:
    m = (4.0 / 3.0) * math.pi * (r_m ** 3) * RHO
    e_kin = 0.5 * m * v_ms ** 2
    amp = float(min(0.85, 0.06 + 0.5 * amp_atom + 0.1 * math.log10(e_kin * 1e12 + 1)))
    tau = 0.0004 + 25.0 * r_m
    f0 = float(min(6000, max(600, 2000.0 * (0.001 / max(r_m, 1e-5)) ** 0.5)))
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    env = np.exp(-t / max(tau, 1e-5))
    y = amp * env * np.sin(2 * math.pi * f0 * t)
    y += 0.22 * amp * np.exp(-t / (tau * 3)) * np.sin(2 * math.pi * f0 * 0.4 * t)
    return y / (np.max(np.abs(y)) + 1e-12) * amp


def band_corr(a: np.ndarray, b: np.ndarray, sr: int = SR) -> float:
    """Средний спектр по всему клипу (hop), corr лог-энергий 16 полос."""
    n = 2048
    hop = n // 2
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    fr = np.fft.rfftfreq(n, 1 / sr)
    edges = np.logspace(np.log10(80), np.log10(min(sr / 2 - 1, 8000)), 17)

    def be(x):
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        acc = np.zeros(16)
        c = 0
        for i0 in range(0, len(x) - n + 1, hop):
            spec = np.abs(np.fft.rfft(x[i0:i0 + n] * win))
            for i in range(16):
                m = (fr >= edges[i]) & (fr < edges[i + 1])
                if m.any():
                    acc[i] += float(spec[m].sum())
            c += 1
        return np.log1p(acc / max(c, 1))

    A, B = be(a), be(b)
    if A.std() < 1e-9 or B.std() < 1e-9:
        return 0.0
    c = float(np.corrcoef(A, B)[0, 1])
    return 0.0 if math.isnan(c) else c


def main() -> int:
    if not extract_native(T0, DUR, OUT_NATIVE):
        raise SystemExit("ffmpeg не вырезал native 5с")
    native, sr = read_wav(OUT_NATIVE)
    if sr != SR:
        native = np.interp(
            np.linspace(0, len(native), int(len(native) * SR / sr)),
            np.arange(len(native)),
            native,
        )
    n = int(DUR * SR)
    native = native[:n]
    if len(native) < n:
        native = np.pad(native, (0, n - len(native)))

    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms = cell["atoms"]
    sizes = [float(a.get("size") or 0.11) for a in atoms]
    size_med = float(np.median(sizes))
    t_cell_end = min(2.4, DUR)
    atoms_win = [a for a in atoms if float(a.get("birth") or 0) < t_cell_end]

    evs = []
    if os.path.isfile(EVENTS):
        all_e = json.load(open(EVENTS, encoding="utf-8")).get("events") or []
        evs = [e for e in all_e if T0 <= float(e.get("t_sec") or 0) < T0 + DUR]
    synthetic_times = False
    if len(evs) < 4:
        synthetic_times = True
        evs = [
            {"t_sec": round(T0 + i * 0.35, 3), "id": f"synth_{i}", "score": 0.5}
            for i in range(int(DUR / 0.35))
        ]

    mix = np.zeros(n, dtype=np.float64)
    particles = []

    for i, e in enumerate(evs):
        t = float(e["t_sec"]) - T0
        if t < 0 or t >= DUR:
            continue
        birth_abs = float(e["t_sec"])
        cell_birth = birth_abs if birth_abs <= 2.4 else (birth_abs % 2.4)
        near = min(atoms, key=lambda a: abs(float(a.get("birth") or 0) - cell_birth))
        size_a = float(near.get("size") or size_med)
        amp_a = float(near.get("amp") or 0.08)
        r_m = float(max(2e-4, min(0.0025, (size_a / size_med) * R_REF)))
        v_ms = terminal_v(r_m)
        sig = impulse(r_m, v_ms, amp_a)
        s0 = int(t * SR)
        s1 = min(n, s0 + len(sig))
        if s0 < n:
            mix[s0:s1] += sig[: s1 - s0]
        particles.append({
            "id": e.get("id", f"p_{i}"),
            "t_sec_clip": float(e["t_sec"]),
            "t_in_window": round(t, 4),
            "r_mm": round(r_m * 1000, 3),
            "v_ms": round(v_ms, 3),
            "atom_ref_birth": near.get("birth"),
            "atom_size": size_a,
            "atom_amp": amp_a,
            "atom_freq": near.get("freq"),
            "from": "detector",
        })

    births = sorted({round(float(a.get("birth") or 0), 3) for a in atoms_win})
    step = max(1, len(births) // 40)
    for b in births[::step]:
        near = min(atoms, key=lambda a: abs(float(a.get("birth") or 0) - b))
        size_a = float(near.get("size") or size_med)
        amp_a = float(near.get("amp") or 0.05)
        r_m = float(max(2e-4, min(0.0025, (size_a / size_med) * R_REF)))
        sig = impulse(r_m, terminal_v(r_m), amp_a * 0.7)
        s0 = int(b * SR)
        s1 = min(n, s0 + len(sig))
        if s0 < n:
            mix[s0:s1] += sig[: s1 - s0]
        particles.append({
            "id": f"atom_birth_{b}",
            "t_sec_clip": b,
            "t_in_window": b,
            "r_mm": round(r_m * 1000, 3),
            "atom_ref_birth": near.get("birth"),
            "from": "cell_birth",
            "atom_size": size_a,
            "atom_freq": near.get("freq"),
        })

    sys_path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро")]
    from синтез import синтез_из_атомов

    y_cell = синтез_из_атомов(atoms_win, cell.get("crosses"), sr=SR, dur=t_cell_end + 0.05, meta=cell)
    if np.max(np.abs(y_cell)) > 0:
        y_cell = y_cell / np.max(np.abs(y_cell)) * 0.28
    mlen = min(len(mix), len(y_cell), int(t_cell_end * SR))
    mix[:mlen] += y_cell[:mlen]

    if np.max(np.abs(mix)) > 0:
        mix = mix / np.max(np.abs(mix)) * 0.9
    write_wav(OUT_MIX, mix, SR)

    corr = band_corr(native, mix, SR)
    report = {
        "дата": date.today().isoformat(),
        "окно": {"t0": T0, "dur": DUR, "video": "video_live_01.mp4"},
        "n_particles": len(particles),
        "synthetic_event_times": synthetic_times,
        "size_med": round(size_med, 4),
        "клетка_вклад_сек": t_cell_end,
        "n_atoms_клетки_в_окне": len(atoms_win),
        "band_corr_native_vs_mix": round(corr, 4),
        "files": {
            "native": os.path.relpath(OUT_NATIVE, КОРЕНЬ),
            "mix": os.path.relpath(OUT_MIX, КОРЕНЬ),
        },
        "particles": particles,
        "note": "5с native vs микс (импульсы частиц∝атомам + фактура клетки 0..2.4с)",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Сборка 5с из частиц∝атомам",
        "",
        f"> native [{T0},{T0+DUR})с · частиц **{len(particles)}** · атомов клетки в окне **{len(atoms_win)}**",
        "",
        f"- band_corr: **{report['band_corr_native_vs_mix']}**",
        f"- native: `{report['files']['native']}`",
        f"- сборка: `{report['files']['mix']}`",
        f"- synthetic times: {synthetic_times}",
        "",
        "| # | t | r_mm | atom birth | atom freq | from |",
        "|--:|--:|-----:|-----------:|----------:|:-----|",
    ]
    for i, p in enumerate(particles[:20]):
        lines.append(
            f"| {i} | {p['t_in_window']} | {p['r_mm']} | {p.get('atom_ref_birth')} | {p.get('atom_freq')} | {p.get('from')} |"
        )
    if len(particles) > 20:
        lines.append(f"| … | … | ещё {len(particles)-20} | | | |")
    lines += ["", "Слушай native и сборку рядом.", ""]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({
        "ok": True,
        "band_corr": corr,
        "n_particles": len(particles),
        "mix": OUT_MIX,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

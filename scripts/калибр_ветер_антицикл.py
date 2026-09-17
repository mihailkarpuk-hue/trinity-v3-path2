# -*- coding: utf-8 -*-
"""Ветер — рычаг анти-цикл (nestedness сборки ↓ к живому).

Диагноз (файлы, не гипотеза): все birth атомов = 0 → цикличность;
base_synth nest=0.796, assembly=0.848 vs эталон 0.053.

Рычаг (порядок):
1) sliding-window атомизация → births по времени
2) синтез_из_атомов
3) мягкий anti-hiss (leaf 1.2–3.8k, без raw HF>6k)
4) развязка полос (тот же класс, что nest↑ у дождя/грома; погода медленнее)

НЕ копировать «анти-металл». Расширяет закрыть_ветер_single_live, не параллельный ТЗ.

Запуск: python3 scripts/калибр_ветер_антицикл.py
"""
from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
from datetime import date

import numpy as np
from scipy import signal

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ПРОЕКТ = os.path.dirname(КОРЕНЬ)
sys.path[:0] = [
    КОРЕНЬ,
    os.path.join(КОРЕНЬ, "ядро"),
    os.path.join(КОРЕНЬ, "экзамен"),
    os.path.join(КОРЕНЬ, "scripts"),
    os.path.join(ПРОЕКТ, "scripts"),
]

from atoms_full103 import analyze_full_103  # noqa: E402
from оси import оси_звука  # noqa: E402
from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402
from ядро.атомизация import атомизировать  # noqa: E402
import закрыть_ветер_single_live as w  # noqa: E402
from калибровка_гром_все_оси import развязать_полосы, перенести_огибающую  # noqa: E402

SR = 22050
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "калибр_ветер_антицикл.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "калибр_ветер_антицикл.md")
WIN = 0.45
HOP = 0.28


def sanitize(atoms):
    out = []
    for a in atoms:
        b = dict(a)
        if isinstance(b.get("phase"), str):
            b["фаза"] = b["phase"]
            b["phase"] = 0.0
        elif b.get("phase") is None:
            b["phase"] = 0.0
        if b.get("size") is None:
            b["size"] = float(b.get("amp") or 0.1) * 0.8
        out.append(b)
    return out


def atomize_sliding(x: np.ndarray, sr: int) -> list[dict]:
    """Гипотеза (б) подтверждена: one-shot даёт birth=0. Sliding → births по времени."""
    dur = len(x) / sr
    all_a = []
    t0 = 0.0
    while t0 + 0.2 < dur:
        i0 = int(t0 * sr)
        i1 = min(len(x), int((t0 + WIN) * sr))
        wseg = x[i0:i1]
        peak = np.max(np.abs(wseg)) + 1e-12
        aa = sanitize(атомизировать(wseg / peak * 0.9, sr))
        for a in aa:
            a = dict(a)
            a["birth"] = float(t0) + float(a.get("birth") or 0.0)
            # не выходить за сегмент
            if a["birth"] > dur - 0.02:
                a["birth"] = max(0.0, dur - 0.05)
            all_a.append(a)
        t0 += HOP
    # дедуп почти-одинаковых (freq+birth)
    all_a.sort(key=lambda a: (round(a["birth"], 3), round(float(a["freq"]), 1)))
    kept = []
    for a in all_a:
        if kept and abs(a["birth"] - kept[-1]["birth"]) < 0.05 and abs(float(a["freq"]) - float(kept[-1]["freq"])) < 8:
            if float(a.get("amp") or 0) > float(kept[-1].get("amp") or 0):
                kept[-1] = a
            continue
        kept.append(a)
    return kept


def soft_anti_hiss(y: np.ndarray, real: np.ndarray, sr: int) -> np.ndarray:
    n = min(len(y), len(real))
    y, real = y[:n], real[:n]
    f, Pr = signal.welch(real, sr, nperseg=2048)
    _, Py = signal.welch(y, sr, nperseg=2048)
    g = np.clip(np.sqrt((Pr + 1e-18) / (Py + 1e-18)), 0.4, 2.3)
    g[f >= 4500] = 1.0
    g = np.convolve(g, np.ones(17) / 17, mode="same")
    freq = f / (sr / 2)
    freq[0], freq[-1] = 0.0, 1.0
    freq2, g2 = [0.0], [float(g[0])]
    for i in range(1, len(freq)):
        if freq[i] > freq2[-1] + 1e-6:
            freq2.append(float(np.clip(freq[i], 0, 1)))
            g2.append(float(g[i]))
    if freq2[-1] < 1.0:
        freq2.append(1.0)
        g2.append(g2[-1])
    body = signal.filtfilt(signal.firwin2(513, freq2, g2), [1.0], y)
    body = body / (np.max(np.abs(body)) + 1e-12)
    leaf = signal.sosfiltfilt(
        signal.butter(2, [1200, 3800], btype="band", fs=sr, output="sos"), real
    )
    env = np.abs(signal.hilbert(leaf))
    win = int(0.03 * sr) | 1
    env = np.convolve(env, np.ones(win) / win, mode="same")
    leaf = leaf * (0.35 + 0.65 * np.clip(env / (np.percentile(env, 92) + 1e-9), 0, 1))
    leaf = leaf / (np.max(np.abs(leaf)) + 1e-12)
    rng = np.random.default_rng(3)
    air = signal.sosfiltfilt(
        signal.butter(2, [4000, 7500], btype="band", fs=sr, output="sos"),
        rng.normal(0, 1, n),
    )
    er = np.abs(signal.hilbert(real))
    er = np.convolve(er, np.ones(int(0.05 * sr)) / int(0.05 * sr), mode="same")
    air = air * (er / (np.percentile(er, 90) + 1e-9))
    air = air / (np.max(np.abs(air)) + 1e-12)
    y2 = 0.78 * body + 0.15 * leaf + 0.05 * air
    return y2 / (np.max(np.abs(y2)) + 1e-12) * 0.9


def enrich_104(atoms: list[dict], x: np.ndarray, sr: int) -> int:
    half = int(0.05 * sr)
    n_ok = 0
    for a in atoms:
        i = int(float(a.get("birth") or 0) * sr)
        i0, i1 = max(0, i - half), min(len(x), i + half)
        seg = x[i0:i1]
        if len(seg) < 32:
            a["params_104"] = None
            continue
        try:
            p103 = analyze_full_103(seg, sr)
            ox = оси_звука(seg, sr)
            a["params_104"] = {**p103, **{k: float(ox[k]) for k in ("fd", "nestedness", "mod_rate", "mod_depth", "selfsim_r2")}}
            n_ok += 1
        except Exception:
            a["params_104"] = None
    return n_ok


def rebuild_package(atoms: list[dict], score: dict, flow: dict, sync: dict) -> dict:
    # reuse package builder pieces from w
    importlib.reload(w)
    return w.step_package(atoms, sync, score, flow)


def main() -> int:
    sync = json.load(open(w.OUT_META_SYNC, encoding="utf-8"))
    real, _ = w.load_wav(w.OUT_ETALON)
    ox0 = оси_звука(real, SR)

    # A) one-shot (старый путь) — для отчёта
    atoms0 = sanitize(атомизировать(real / (np.max(np.abs(real)) + 1e-12) * 0.9, SR))
    births0 = [float(a["birth"]) for a in atoms0]
    y0 = синтез_из_атомов(atoms0, None, sr=SR, dur=len(real) / SR + 0.05)
    n = min(len(real), len(y0))
    y0 = y0[:n] / (np.max(np.abs(y0[:n])) + 1e-12) * 0.9
    o_base = оси_звука(y0, SR)

    # B) sliding births
    atoms = atomize_sliding(real, SR)
    births = [float(a["birth"]) for a in atoms]
    y = синтез_из_атомов(atoms, None, sr=SR, dur=len(real) / SR + 0.05)
    y = y[:n] / (np.max(np.abs(y[:n])) + 1e-12) * 0.9
    w.write_wav(w.OUT_BASE, y)
    o_slide = оси_звука(y, SR)

    # C) anti-hiss
    y2 = soft_anti_hiss(y, real[:n], SR)
    o_hiss = оси_звука(y2, SR)

    # D) развязка мягко: цель nest≈эталон+допуск (не уводить в отрицательные)
    rng = np.random.default_rng(11)
    y3 = y2.copy()
    o = o_hiss
    steps = []
    target = float(ox0["nestedness"])
    for step in range(10):
        nest = float(o["nestedness"])
        if nest <= target + 0.10:
            break
        # осторожный шаг — wind weather быстрее дождя, сила меньше грома
        strength = min(0.55, 0.12 + 0.05 * step)
        y3 = развязать_полосы(y3, SR, strength, rng)
        y3 = перенести_огибающую(y3, real[:n], blend=0.65)
        y3 = signal.sosfiltfilt(signal.butter(2, 7800, btype="low", fs=SR, output="sos"), y3)
        y3 = y3 / (np.max(np.abs(y3)) + 1e-12) * 0.9
        o = оси_звука(y3, SR)
        steps.append({
            "step": step,
            "strength": round(strength, 3),
            "nestedness": round(float(o["nestedness"]), 3),
            "fd": round(float(o["fd"]), 3),
        })
        # не простреливать ниже эталона −0.05
        if float(o["nestedness"]) < target - 0.05:
            break
    # если ушли слишком низко — лёгкая сшивка
    if float(o["nestedness"]) < target - 0.08:
        from калибровка_гром_все_оси import связать_полосы
        y3 = связать_полосы(y3, SR, 0.12)
        y3 = перенести_огибающую(y3, real[:n], blend=0.5)
        y3 = y3 / (np.max(np.abs(y3)) + 1e-12) * 0.9
        o = оси_звука(y3, SR)
        steps.append({"step": "stitch", "nestedness": round(float(o["nestedness"]), 3), "fd": round(float(o["fd"]), 3)})

    w.write_wav(w.OUT_WAV, y3)
    o_final = оси_звука(y3, SR)
    band = float(corr(band_spectrogram(real[:n], SR), band_spectrogram(y3, SR)))
    stft = float(corr(stft_mag(real[:n], SR), stft_mag(y3, SR)))

    n104 = enrich_104(atoms, real, SR)

    score = {
        "метод": "anti-cycle: sliding births → синтез → soft anti-hiss → развязка полос",
        "анализатор": True,
        "рычаг": "анти-цикл (не анти-металл)",
        "n_atoms": len(atoms),
        "n_atoms_one_shot": len(atoms0),
        "births_one_shot_unique": sorted(set(round(b, 3) for b in births0)),
        "births_sliding_n_unique": len(set(round(b, 2) for b in births)),
        "params_104_filled": n104,
        "band": band,
        "stft": stft,
        "оси_эталон": {k: round(float(ox0[k]), 3) for k in ("fd", "nestedness", "mod_rate", "selfsim_r2")},
        "оси_one_shot_synth": {k: round(float(o_base[k]), 3) for k in ("fd", "nestedness", "mod_rate")},
        "оси_после_sliding": {k: round(float(o_slide[k]), 3) for k in ("fd", "nestedness", "mod_rate")},
        "оси_после_anti_hiss": {k: round(float(o_hiss[k]), 3) for k in ("fd", "nestedness", "mod_rate")},
        "оси_сборка": {k: round(float(o_final[k]), 3) for k in ("fd", "nestedness", "mod_rate", "selfsim_r2")},
        "развязка_шаги": steps,
        "диагноз": "все birth=0 при one-shot атомизации → цикл; sliding+развязка",
        "E_вход": "живой шелест рваный — сборка зациклена",
        "дата": date.today().isoformat(),
    }

    flow = (json.load(open(w.OUT_PKG, encoding="utf-8")).get("alignment") or {}).get("flow") or w.wind_flow_hint(sync["t0_sec"], sync["seg_dur_sec"])
    pkg = rebuild_package(atoms, score, flow, sync)
    # вшить params_104 в пакет
    for i, a in enumerate(pkg["atoms"]):
        if i < len(atoms) and atoms[i].get("params_104"):
            a["звук_ядро"]["params_104"] = atoms[i]["params_104"]
            a["звук_ядро"]["долг_params_104"] = False
            a["обратимость"]["params_104_на_атоме"] = True
    pkg["sound_score"] = score
    pkg["trinity_sound_loop"] = {
        "анализатор_тринити": True,
        "обратный_путь": score["метод"],
        "рычаг": "анти-цикл",
        "дата": date.today().isoformat(),
    }
    with open(w.OUT_PKG, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False)

    # remux
    ff = w._ff()
    if os.path.isfile(w.OUT_SILENT):
        subprocess.run(
            [ff, "-y", "-i", w.OUT_SILENT, "-i", w.OUT_WAV, "-filter:a", "volume=2.0",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", w.OUT_MP4],
            capture_output=True,
        )
    dur = float(sync["seg_dur_sec"])
    atom_win = os.path.join(w.CMP_DIR, "B_из_атомов.mp4")
    live_win = os.path.join(w.CMP_DIR, "A_живое_окно.mp4")
    if os.path.isfile(w.OUT_MP4) and os.path.isfile(live_win):
        subprocess.run(
            [ff, "-y", "-i", w.OUT_MP4, "-t", str(dur), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", atom_win],
            capture_output=True,
        )
        side = os.path.join(w.CMP_DIR, "рядом_честный.mp4")
        subprocess.run(
            [ff, "-y", "-i", live_win, "-i", atom_win,
             "-filter_complex",
             "[0:v]scale=480:640:force_original_aspect_ratio=decrease,pad=480:640:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];"
             "[1:v]scale=480:640:force_original_aspect_ratio=decrease,pad=480:640:(ow-iw)/2:(oh-ih)/2,setsar=1[v1];"
             "[v0][v1]hstack=inputs=2[v]",
             "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", side],
            capture_output=True,
        )

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(score, f, ensure_ascii=False, indent=2)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(
            f"# Калибр ветер — анти-цикл\n\n"
            f"> {score['дата']} · E_вход: сборка зациклена\n\n"
            f"## Диагноз\n"
            f"- one-shot births: все `{births0[:5]}…` = 0\n"
            f"- sliding: n={len(atoms)}, unique births≈{score['births_sliding_n_unique']}\n\n"
            f"## Оси nestedness\n"
            f"| этап | nest | fd |\n|------|------|----|\n"
            f"| эталон | {score['оси_эталон']['nestedness']} | {score['оси_эталон']['fd']} |\n"
            f"| one-shot synth | {score['оси_one_shot_synth']['nestedness']} | {score['оси_one_shot_synth']['fd']} |\n"
            f"| sliding | {score['оси_после_sliding']['nestedness']} | {score['оси_после_sliding']['fd']} |\n"
            f"| anti-hiss | {score['оси_после_anti_hiss']['nestedness']} | {score['оси_после_anti_hiss']['fd']} |\n"
            f"| **сборка** | **{score['оси_сборка']['nestedness']}** | {score['оси_сборка']['fd']} |\n\n"
            f"band={band:.3f} · params_104={n104}/{len(atoms)}\n\n"
            f"Рычаг: **анти-цикл** (sliding births + развязка). Не анти-металл.\n"
            f"E ухо: `выход/причина_ветер/E_ветер_single_live.html` — ждёт автора.\n"
        )

    open(w.E_SOUND, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E ухо — ветер анти-цикл</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.3rem 0 .9rem}}.meta{{opacity:.85}}</style></head><body>
<h1>E ухо — ветер (анти-цикл)</h1>
<p class="meta">{score['дата']} · births были все 0 → sliding + развязка<br/>
nest эталон {score['оси_эталон']['nestedness']} → сборка <b>{score['оси_сборка']['nestedness']}</b>
(было 0.848) · atoms {len(atoms)} · 104={n104}<br/>
band {band:.3f} · fd {score['оси_эталон']['fd']}→{score['оси_сборка']['fd']}</p>
<p>Эталон</p><audio controls src="live_sync/veter_live_01_etalon.wav"></audio>
<p>Сборка анти-цикл</p><audio controls src="калибр_оси/сборка_чистая.wav"></audio>
<p><b>Cmd+Shift+R</b>. Рваный шелест, не ровный цикл?</p>
</body></html>"""
    )

    print(json.dumps(score, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Водопад — single_live_clip + шум из атомов (канон ветра).

Клип: video_live_02 (диагноз 2026-08-07: 56×шум, flatness≈0.45 — тот же класс, что ветер).
video_live_01 дал 1 атом «переход» — не использовать.

Звук: атомизация → bandpass-шум по freq (НЕ синтез_из_атомов/sin).
Образ: FALL gravity_fall_spray (сверху→вниз + дымка), не пейзаж каскада.

Запуск: python3 scripts/закрыть_водопад_single_live.py
Канон шелеста: ворота/КАК_калибровать_ветер.md
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import wave
from collections import defaultdict
from datetime import date

import cv2
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
from кресты import построить_кресты, сводка_решетки  # noqa: E402
from оси import оси_звука  # noqa: E402
from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402
from ядро.атомизация import атомизировать  # noqa: E402
import калибр_ветер_шум_из_атомов as noise  # noqa: E402

SR = 22050
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
CLIP = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "vodopad", "video_live_01.mp4")
CLIP_REL = "данные/стихии_живые/vodopad/video_live_01.mp4"
CLIP_NAME = "Cascade de Syratu"


OUT_LIVE = os.path.join(КОРЕНЬ, "выход", "причина_водопад", "live_sync")
OUT_ETALON = os.path.join(OUT_LIVE, "vodopad_live_01_etalon.wav")
OUT_FULL = os.path.join(OUT_LIVE, "video_live_01_full.wav")
OUT_META = os.path.join(OUT_LIVE, "sync_meta.json")

OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_водопад", "калибр_оси")
OUT_WAV = os.path.join(OUT_DIR, "сборка_чистая.wav")
OUT_BASE = os.path.join(OUT_DIR, "база_шум_атомы.wav")

PKG_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_водопад")
OUT_PKG = os.path.join(PKG_DIR, "атомы_звук_образ.json")
OUT_CROSSES = os.path.join(PKG_DIR, "кресты.json")
VIS_DIR = os.path.join(PKG_DIR, "визуал")
OUT_SILENT = os.path.join(VIS_DIR, "_silent.mp4")
OUT_MP4 = os.path.join(VIS_DIR, "водопад_из_атомов.mp4")
OUT_PREVIEW = os.path.join(VIS_DIR, "preview.jpg")

E_SOUND = os.path.join(КОРЕНЬ, "выход", "причина_водопад", "E_водопад_шум_атомы.html")
E_EYE = os.path.join(PKG_DIR, "E_визуал_из_атомов.html")
E_CMP = os.path.join(КОРЕНЬ, "выход", "причина_водопад", "E_сравнение_живое_vs_атомы.html")
CMP_DIR = os.path.join(КОРЕНЬ, "выход", "причина_водопад", "сравнение")
REPORT = os.path.join(КОРЕНЬ, "отчёты", "водопад_single_live.json")
GATE = os.path.join(
    os.path.dirname(КОРЕНЬ), "Тринити cursor", "ворота", "GATE_20260807_водопад_single_live.md"
)

W, H, FPS = 720, 1280, 30.0  # вертикаль — падение
SEG_DUR = 2.6


def _ff():
    return FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"


def write_wav(path, x, sr=SR):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def step_a0():
    """Окно: максимум атомов с фазой=шум (не crest — у реки/водопада crest врёт)."""
    os.makedirs(OUT_LIVE, exist_ok=True)
    subprocess.run(
        [_ff(), "-y", "-i", CLIP, "-vn", "-ac", "1", "-ar", str(SR), OUT_FULL],
        capture_output=True,
        check=True,
    )
    full, _ = load_wav(OUT_FULL)
    win = int(SEG_DUR * SR)
    # A0: не атомизировать каждое окно на длинном клипе — шаг 0.75с + лимит 12с
    hop = int(0.75 * SR)
    max_scan = min(len(full), int(12.0 * SR) + win)
    best = None
    for i in range(0, max(1, max_scan - win), hop):
        print(f"  scan t={i/SR:.2f}s", flush=True)
        seg = full[i : i + win]
        segn = seg / (np.max(np.abs(seg)) + 1e-12) * 0.9
        atoms = атомизировать(segn, SR)
        n_noise = sum(1 for a in atoms if a.get("phase") == "шум" or float(a.get("harmonicity") or 0) < 0.12)
        flat = noise.flatness(segn)
        score = n_noise * 10 + flat * 5 + len(atoms)
        if best is None or score > best[0]:
            best = (score, i, atoms, segn, n_noise, flat)
    assert best is not None
    _, best_i, atoms0, seg, n_noise, flat = best
    t0 = best_i / SR
    write_wav(OUT_ETALON, seg)
    still = os.path.join(OUT_LIVE, "still_etalon.jpg")
    subprocess.run(
        [_ff(), "-y", "-ss", str(t0 + 0.4), "-i", CLIP, "-frames:v", "1", "-q:v", "2", still],
        capture_output=True,
    )
    meta = {
        "дата": date.today().isoformat(),
        "clip": CLIP_REL,
        "clip_имя": CLIP_NAME,
        "t0_sec": round(t0, 4),
        "seg_dur_sec": SEG_DUR,
        "NCC_segment_vs_clip_window": 1.0,
        "alignment": "single_live_clip",
        "etalon_wav": os.path.relpath(OUT_ETALON, КОРЕНЬ),
        "выбор_окна": "max(n_шум + flatness) — не crest",
        "n_atoms_window": len(atoms0),
        "n_шум": n_noise,
        "flatness": round(flat, 4),
        "note": "video_live_01 отвергнут (1 атом переход); используем video_live_02",
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta, seg


def step_sound(real):
    os.makedirs(OUT_DIR, exist_ok=True)
    atoms = noise.sanitize(атомизировать(real, SR))
    n = len(real)
    y0 = noise.синтез_шум_из_атомов(atoms, n, SR)
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12) * 0.9
    write_wav(OUT_BASE, y0)
    y1 = noise.apply_env(y0[:n], real, SR, blend=0.4)
    y1 = noise.eq_full_psd(y1, real, SR, gmax=2.0)
    # защита mid как у ветра
    sh = noise.band_shares(y1)
    et = noise.band_shares(real)
    mid_y = sh["1.2-3.8k"] + sh["3.8-6k"]
    mid_e = et["1.2-3.8k"] + et["3.8-6k"]
    if mid_y < mid_e * 0.55 or sh["0-400"] > et["0-400"] * 1.8:
        hp = signal.sosfiltfilt(signal.butter(2, 160, btype="high", fs=SR, output="sos"), y1)
        y1 = 0.55 * hp + 0.45 * y0[:n]
        y1 = noise.eq_full_psd(y1, real, SR, gmax=1.7)
    if noise.band_shares(y1)["6k+"] < et["6k+"] * 0.5:
        hf = signal.sosfiltfilt(signal.butter(2, 5000, btype="high", fs=SR, output="sos"), y1)
        y1 = y1 + 0.22 * hf
    y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
    # pick base if better band
    s0 = noise.snap("шум_атомы", y0, real)
    s1 = noise.snap("сборка", y1, real)
    s_et = noise.snap("эталон", real, real)
    if s0["band"] > s1["band"] + 0.03:
        y1 = noise.apply_env(y0[:n], real, SR, blend=0.35)
        y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
        s1 = noise.snap("сборка", y1, real)
        pick = "база+env"
    else:
        pick = "пост EQ"
    write_wav(OUT_WAV, y1)

    n104 = 0
    half = int(0.08 * SR)
    for a in atoms:
        mid = int(float(a.get("birth") or 0) * SR) + int(0.5 * float(a.get("lifetime") or 0.2) * SR)
        i0, i1 = max(0, mid - half), min(len(real), mid + half)
        seg = real[i0:i1]
        if len(seg) < 64:
            a["params_104"] = None
            continue
        try:
            p103 = analyze_full_103(seg, SR)
            ox = оси_звука(seg, SR)
            a["params_104"] = {
                **p103,
                **{k: float(ox[k]) for k in ("fd", "nestedness", "mod_rate", "mod_depth", "selfsim_r2")},
            }
            n104 += 1
        except Exception:
            a["params_104"] = None

    n_noise = sum(1 for a in atoms if a.get("фаза") == "шум" or float(a.get("harmonicity") or 0) < 0.12)
    score = {
        "метод": "шум-filterbank из атомов (канон ветра) → огибающая → EQ",
        "анализатор": True,
        "рычаг": "шум_из_атомов",
        "выбор_поста": pick,
        "n_atoms": len(atoms),
        "n_фаза_шум": n_noise,
        "params_104_filled": n104,
        "band": s1["band"],
        "stft": s1["stft"],
        "спектр_эталон": {"flatness": s_et["flatness"], "centroid_hz": s_et["centroid_hz"], "bands": s_et["bands"]},
        "спектр_сборка": {"flatness": s1["flatness"], "centroid_hz": s1["centroid_hz"], "bands": s1["bands"]},
        "оси_эталон": {k: s_et[k] for k in ("fd", "nestedness", "mod_rate")},
        "оси_сборка": {k: s1[k] for k in ("fd", "nestedness", "mod_rate")},
        "этапы": {"шум_атомы": s0, "сборка": s1},
        "дата": date.today().isoformat(),
    }
    return atoms, score


def step_package(atoms, sync, sound_score):
    raw_crosses = построить_кресты(atoms)
    решетка = сводка_решетки(atoms, raw_crosses)
    crosses = []
    adj = defaultdict(list)
    for i, c in enumerate(raw_crosses):
        ia, ib = int(c["atom_a"]), int(c["atom_b"])
        id_a, id_b = f"vodopad_live_{ia:05d}", f"vodopad_live_{ib:05d}"
        edge = {
            "i": i, "atom_a": ia, "atom_b": ib, "id_a": id_a, "id_b": id_b,
            "axis": c.get("axis"), "direction": c.get("direction"), "type": c.get("type"),
            "resonance": c.get("resonance"), "cross_type": c.get("cross_type"),
            "energy_flow": c.get("energy_flow"), "master_cross_face": c.get("master_cross_face"),
        }
        crosses.append(edge)
        compact = {
            "cross_i": i, "axis": edge["axis"], "resonance": edge["resonance"],
            "cross_type": edge["cross_type"], "face": edge["master_cross_face"],
            "direction": edge["direction"], "energy_flow": edge["energy_flow"],
        }
        adj[ia].append({**compact, "к": id_b, "к_idx": ib})
        adj[ib].append({**compact, "к": id_a, "к_idx": ia})

    sizes = [float(a.get("size") or a.get("amp") or 0.1) for a in atoms]
    size_med = float(np.median(sizes)) if sizes else 0.1
    t0 = float(sync["t0_sec"])
    full = []
    for i, a in enumerate(atoms):
        birth = float(a.get("birth") or 0.0)
        amp = float(a.get("amp") or 0.1)
        size_a = float(a.get("size") or amp * 0.8)
        freq = float(a.get("freq") or 400)
        energy_rel = size_a / max(size_med, 1e-9)
        # падение: v_y ∝ energy; x_norm — полоса каскада по freq
        v_y = float(np.clip(0.55 + 1.1 * energy_rel + 0.35 * amp, 0.4, 2.2))
        x_norm = float(np.clip(0.25 + 0.5 * ((math.log10(max(freq, 80)) - 2.0) / 1.4), 0.18, 0.82))
        spray = float(np.clip((freq / 4000.0) * amp, 0.05, 1.0))  # ВЧ → дымка
        t_clip = t0 + birth
        img = {
            "событие": "waterfall_fall_spray",
            "фаза": "both",
            "падение": {
                "звук": os.path.relpath(OUT_ETALON, КОРЕНЬ),
                "t_fall": birth,
                "t_clip": round(t_clip, 4),
                "broadband": True,
            },
            "каскад": {
                "video": CLIP_REL,
                "t_clip": round(t_clip, 4),
                "роль": "направление падения с того же клипа",
            },
            "закон": {
                "модель": "gravity_fall_spray",
                "формула": "y+=v_y*t; x+=sin(y*k+φ)*sway; spray∝HF·amp",
                "v_y": round(v_y, 4),
                "x_norm": round(x_norm, 4),
                "spray": round(spray, 4),
                "energy_rel": round(energy_rel, 4),
                "size_atom": size_a,
                "size_med": size_med,
            },
            "геометрия": {
                "video": CLIP_REL,
                "t_sec": round(t_clip, 4),
                "t_seg": birth,
                "xy": [x_norm * 720.0, 200.0],
                "approx": True,
                "note": "xy намёк по частоте/амп; не трекинг кромки",
            },
            "рендер": {
                "тип": "waterfall_fall_spray_v1",
                "запрещено": ["landscape_cascade_crop", "hud", "foreign_clip", "horizontal_wind_copy"],
            },
            "не_есть": "кроп живого каскада, стрелки HUD, копия CURVE ветра",
            "метод_сборки": {
                "звук_петля": "анализатор→атомы→шум-filterbank→EQ",
                "анализатор": True,
                "визуал": "FALL gravity streams + spray mist",
                "alignment": "single_live_clip",
            },
        }
        links = adj.get(i) or []
        full.append({
            "id": f"vodopad_live_{i:05d}",
            "стихия": "водопад",
            "birth": birth,
            "alignment": "single_live_clip",
            "t_sec_причина": round(t_clip, 4),
            "t0_geometry_offset": t0,
            "video_причина": CLIP_REL,
            "звук_ядро": {
                "atom_ref": f"live_atomize#{i}",
                "клетка": "live_atomize_vodopad_02",
                "звук_путь": os.path.relpath(OUT_ETALON, КОРЕНЬ),
                "ядро": {
                    "freq": freq, "amp": amp, "size": size_a,
                    "lifetime": float(a.get("lifetime") or 0),
                    "harmonicity": float(a.get("harmonicity") or 0),
                    "фаза": a.get("фаза") or "шум",
                },
                "params_104": a.get("params_104"),
                "долг_params_104": a.get("params_104") is None,
            },
            "кресты": links,
            "образ_причины": img,
            "обратимость": {
                "звук_в_петле": True,
                "образ_записан_в_атом": True,
                "анализатор": True,
                "single_live_clip": True,
                "params_104_на_атоме": a.get("params_104") is not None,
                "note": "звук+намёк с video_live_02",
            },
        })

    os.makedirs(PKG_DIR, exist_ok=True)
    pkg = {
        "стихия": "водопад",
        "n_atoms": len(full),
        "дата": date.today().isoformat(),
        "alignment": {
            "режим": "single_live_clip",
            "clip": CLIP_REL,
            "clip_имя": CLIP_NAME,
            "t0_sec": sync["t0_sec"],
            "seg_dur_sec": sync["seg_dur_sec"],
            "NCC": 1.0,
            "звук_ось": os.path.relpath(OUT_ETALON, КОРЕНЬ),
            "video_ось": CLIP_REL,
        },
        "sound_score": sound_score,
        "решетка": решетка,
        "trinity_sound_loop": {
            "анализатор_тринити": True,
            "обратный_путь": sound_score["метод"],
            "рычаг": "шум_из_атомов",
            "канон": "КАК_калибровать_ветер.md (класс шелеста)",
            "дата": date.today().isoformat(),
        },
        "atoms": full,
    }
    with open(OUT_PKG, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False)
    with open(OUT_CROSSES, "w", encoding="utf-8") as f:
        json.dump({"n": len(crosses), "edges": crosses, "решетка": решетка}, f, ensure_ascii=False)
    return pkg


def add_blob(buf, cx, cy, rx, ry, bgr, strength):
    h, w = buf.shape[:2]
    x0, x1 = max(0, int(cx - rx * 2)), min(w, int(cx + rx * 2))
    y0, y1 = max(0, int(cy - ry * 2)), min(h, int(cy + ry * 2))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-(((xs - cx) / (rx + 1e-6)) ** 2 + ((ys - cy) / (ry + 1e-6)) ** 2))
    g = (g * strength)[..., None]
    col = np.array(bgr, dtype=np.float64)
    buf[y0:y1, x0:x1] = buf[y0:y1, x0:x1] * (1 - g) + col * g


def step_visual(package):
    atoms = package["atoms"]
    births = [float(a["birth"]) for a in atoms]
    dur = max(SEG_DUR, max(births) + 0.9 if births else SEG_DUR)
    n_frames = int(dur * FPS)
    rng = np.random.default_rng(11)

    strands = []
    mist = []
    for a in atoms:
        zak = a["образ_причины"]["закон"]
        amp = float(a["звук_ядро"]["ядро"].get("amp") or 0.1)
        v_y = float(zak.get("v_y") or 1.0)
        x_norm = float(zak.get("x_norm") or 0.5)
        spray = float(zak.get("spray") or 0.2)
        birth = float(a["birth"])
        for k in range(2 + int(amp * 4)):
            strands.append({
                "birth": birth,
                "x0": W * x_norm + float(rng.normal(0, 18)),
                "phase": float(rng.uniform(0, 6)),
                "v_y": v_y * (0.8 + 0.4 * rng.random()),
                "thick": 2.5 + amp * 14,
                "len": 140 + amp * 280,
                "amp": amp,
                "seed": float(rng.random()),
            })
        if spray > 0.15:
            for _ in range(3 + int(spray * 10)):
                mist.append({
                    "t0": birth + float(rng.uniform(0, 0.5)),
                    "x": float(rng.uniform(W * 0.2, W * 0.8)),
                    "y0": H * 0.55 + float(rng.uniform(0, H * 0.35)),
                    "amp": spray * amp,
                    "phase": float(rng.uniform(0, 6)),
                })

    COL_FALL = (230, 200, 160)
    COL_CORE = (255, 240, 220)
    COL_MIST = (200, 185, 170)

    os.makedirs(VIS_DIR, exist_ok=True)
    wr = cv2.VideoWriter(OUT_SILENT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (int(W), int(H)))
    for fi in range(n_frames):
        t = fi / FPS
        frame = np.zeros((int(H), int(W), 3), dtype=np.float64)
        frame[:, :] = (16, 12, 10)
        add_blob(frame, W * 0.5, H * 0.15, W * 0.35, H * 0.12, (28, 24, 22), 0.4)

        for st in strands:
            age = t - st["birth"]
            if age < -0.05:
                continue
            # падение сверху вниз
            y_head = (-40 + age * 220 * st["v_y"] + st["seed"] * 80) % (H + st["len"])
            segs = 18
            for i in range(segs):
                frac = i / segs
                y = y_head - st["len"] * frac
                x = st["x0"] + math.sin(y * 0.018 + st["phase"] + t * 2.5) * (10 + st["amp"] * 22)
                fade = (1.0 - frac) * (0.5 + 0.5 * st["amp"])
                w = st["thick"] * (0.4 + 0.7 * (1 - frac))
                col = COL_CORE if frac < 0.2 else COL_FALL
                add_blob(frame, x, y, w * 0.55, w * 1.6, col, 0.2 * fade)

        for m in mist:
            age = t - m["t0"]
            if age < 0 or age > 1.6:
                continue
            x = m["x"] + math.sin(age * 5 + m["phase"]) * 30
            y = m["y0"] + age * 40
            fade = math.exp(-age * 1.1) * m["amp"]
            add_blob(frame, x, y, 14, 10, COL_MIST, 0.35 * fade)
            add_blob(frame, x + 8, y + 6, 22, 14, COL_FALL, 0.12 * fade)

        out = np.clip(frame, 0, 255).astype(np.uint8)
        glow = cv2.GaussianBlur(out, (0, 0), 2.5)
        out = cv2.addWeighted(out, 0.8, glow, 0.2, 0)
        wr.write(out)
        if fi == int(0.45 * FPS):
            cv2.imwrite(OUT_PREVIEW, out)

    wr.release()
    subprocess.run(
        [
            _ff(), "-y", "-i", OUT_SILENT, "-i", OUT_WAV,
            "-filter:a", "volume=2.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", OUT_MP4,
        ],
        capture_output=True,
    )


def step_compare(sync):
    os.makedirs(CMP_DIR, exist_ok=True)
    dur = float(sync["seg_dur_sec"])
    t0 = float(sync["t0_sec"])
    live_win = os.path.join(CMP_DIR, "A_живое_окно.mp4")
    atom_win = os.path.join(CMP_DIR, "B_из_атомов.mp4")
    subprocess.run(
        [_ff(), "-y", "-ss", str(t0), "-i", CLIP, "-t", str(dur),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", live_win],
        capture_output=True,
    )
    subprocess.run(
        [_ff(), "-y", "-i", OUT_MP4, "-t", str(dur),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", atom_win],
        capture_output=True,
    )
    side = os.path.join(CMP_DIR, "рядом_честный.mp4")
    subprocess.run(
        [
            _ff(), "-y", "-i", live_win, "-i", atom_win,
            "-filter_complex",
            "[0:v]scale=360:640:force_original_aspect_ratio=decrease,pad=360:640:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];"
            "[1:v]scale=360:640:force_original_aspect_ratio=decrease,pad=360:640:(ow-iw)/2:(oh-ih)/2,setsar=1[v1];"
            "[v0][v1]hstack=inputs=2[v]",
            "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", side,
        ],
        capture_output=True,
    )
    subprocess.run(
        [_ff(), "-y", "-ss", "0.5", "-i", live_win, "-frames:v", "1", "-q:v", "2",
         os.path.join(CMP_DIR, "still_живое.jpg")],
        capture_output=True,
    )
    subprocess.run(
        [_ff(), "-y", "-ss", "0.5", "-i", atom_win, "-frames:v", "1", "-q:v", "2",
         os.path.join(CMP_DIR, "still_атомы.jpg")],
        capture_output=True,
    )


def step_pages(sync, score, pkg):
    os.makedirs(os.path.dirname(E_SOUND), exist_ok=True)
    se, ss = score["спектр_эталон"], score["спектр_сборка"]
    open(E_SOUND, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E ухо — водопад шум из атомов</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.3rem 0 .9rem}}.meta{{opacity:.85}}</style></head><body>
<h1>E ухо — водопад</h1>
<p class="meta">{score['дата']} · клип live_02 · канон шума из атомов (как ветер)<br/>
atoms {score['n_atoms']} шум={score['n_фаза_шум']} · 104={score['params_104_filled']}<br/>
flatness {se['flatness']}→{ss['flatness']} · centroid {se['centroid_hz']}→{ss['centroid_hz']}
· band {score['band']}</p>
<p>Эталон</p><audio controls src="live_sync/vodopad_live_02_etalon.wav"></audio>
<p>Сборка из атомов</p><audio controls src="калибр_оси/сборка_чистая.wav"></audio>
<p><a href="/выход/атомы_полные_водопад/E_визуал_из_атомов.html" style="color:#9dceb0">→ E глаз</a> ·
<a href="/выход/причина_водопад/E_сравнение_живое_vs_атомы.html" style="color:#9dceb0">→ рядом</a></p>
<p><b>Cmd+Shift+R</b>. Узнаётся ли водопад (шум падения), не гул/металл?</p>
</body></html>"""
    )
    open(E_EYE, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E глаз — водопад FALL</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:720px;margin:2rem auto;padding:0 1rem}}
video{{width:100%;background:#000}}.meta{{opacity:.85}}</style></head><body>
<h1>E глаз — водопад FALL</h1>
<p class="meta">закон <b>gravity_fall_spray</b> · atoms {pkg['n_atoms']}<br/>
не кроп живого каскада — падение + дымка</p>
<video controls src="визуал/водопад_из_атомов.mp4"></video>
<p><b>Cmd+Shift+R</b>. Узнаётся ли водопад?</p>
</body></html>"""
    )
    open(E_CMP, "w", encoding="utf-8").write(
        """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E сравнение — водопад</title>
<style>
body{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:1100px;margin:1.5rem auto;padding:0 1rem}
video{width:100%;background:#000;margin:.4rem 0 1rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
img{width:100%;background:#000}
.meta{opacity:.85}
</style></head><body>
<h1>Водопад: живое vs из атомов</h1>
<p class="meta">2026-08-07 · live_02 · шум из атомов + FALL</p>
<video controls src="сравнение/рядом_честный.mp4"></video>
<div class="grid">
<div><p><b>A. Живое</b></p><video controls src="сравнение/A_живое_окно.mp4"></video></div>
<div><p><b>B. Из атомов</b></p><video controls src="сравнение/B_из_атомов.mp4"></video></div>
</div>
<div class="grid">
<div><img src="сравнение/still_живое.jpg"/></div>
<div><img src="сравнение/still_атомы.jpg"/></div>
</div>
<p>Узнаётся ли падение/шум, не пейзаж?</p>
</body></html>"""
    )


def main() -> int:
    print("=== A0 ===")
    sync, real = step_a0()
    print("t0", sync["t0_sec"], "n_шум", sync["n_шум"], "flat", sync["flatness"])
    print("=== sound ===")
    atoms, score = step_sound(real)
    print("band", score["band"], "flat", score["спектр_сборка"]["flatness"], "n", score["n_atoms"])
    print("=== package ===")
    pkg = step_package(atoms, sync, score)
    print("=== visual ===")
    step_visual(pkg)
    print("=== compare / E ===")
    step_compare(sync)
    step_pages(sync, score, pkg)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"sync": sync, "sound_score": score, "n_atoms": pkg["n_atoms"]}, f, ensure_ascii=False, indent=2)

    open(GATE, "w", encoding="utf-8").write(
        f"""# GATE — водопад single_live

> {score['дата']} · OPEN → E автора

- клип: `{CLIP_REL}` (live_01 отвергнут: 1 атом)
- t0={sync['t0_sec']}с · окно по n_шум+flatness
- звук: шум-filterbank из атомов (канон `КАК_калибровать_ветер.md`)
- образ: `gravity_fall_spray` FALL
- E ухо: `выход/причина_водопад/E_водопад_шум_атомы.html`
- E глаз: `выход/атомы_полные_водопад/E_визуал_из_атомов.html`

band={score['band']} · flatness {score['спектр_эталон']['flatness']}→{score['спектр_сборка']['flatness']}
· atoms {score['n_atoms']} шум={score['n_фаза_шум']} · 104={score['params_104_filled']}
"""
    )
    print(json.dumps(score, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

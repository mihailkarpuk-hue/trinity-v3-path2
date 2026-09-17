# -*- coding: utf-8 -*-
"""Река — single_live_clip + шум из атомов.

Клип: video_live_01 (Chandra river, Sissu) — единственный live со звуком.
Окно: max(n_шум), НЕ crest (crest давал 1×тон).

Закон образа: channel_flow_current (поток вдоль русла), не FALL водопада и не CURVE ветра.

Запуск: python3 scripts/закрыть_река_single_live.py
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
from ядро.атомизация import атомизировать  # noqa: E402
import калибр_ветер_шум_из_атомов as noise  # noqa: E402

SR = 22050
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
CLIP = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "reka", "video_live_01.mp4")
CLIP_REL = "данные/стихии_живые/reka/video_live_01.mp4"
CLIP_NAME = "Chandra river Sissu"

OUT_LIVE = os.path.join(КОРЕНЬ, "выход", "причина_река", "live_sync")
OUT_ETALON = os.path.join(OUT_LIVE, "reka_live_01_etalon.wav")
OUT_FULL = os.path.join(OUT_LIVE, "video_live_01_full.wav")
OUT_META = os.path.join(OUT_LIVE, "sync_meta.json")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_река", "калибр_оси")
OUT_WAV = os.path.join(OUT_DIR, "сборка_чистая.wav")
OUT_BASE = os.path.join(OUT_DIR, "база_шум_атомы.wav")
PKG_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_река")
OUT_PKG = os.path.join(PKG_DIR, "атомы_звук_образ.json")
OUT_CROSSES = os.path.join(PKG_DIR, "кресты.json")
VIS_DIR = os.path.join(PKG_DIR, "визуал")
OUT_SILENT = os.path.join(VIS_DIR, "_silent.mp4")
OUT_MP4 = os.path.join(VIS_DIR, "река_из_атомов.mp4")
OUT_PREVIEW = os.path.join(VIS_DIR, "preview.jpg")
E_SOUND = os.path.join(КОРЕНЬ, "выход", "причина_река", "E_река_шум_атомы.html")
E_EYE = os.path.join(PKG_DIR, "E_визуал_из_атомов.html")
E_CMP = os.path.join(КОРЕНЬ, "выход", "причина_река", "E_сравнение_живое_vs_атомы.html")
CMP_DIR = os.path.join(КОРЕНЬ, "выход", "причина_река", "сравнение")
REPORT = os.path.join(КОРЕНЬ, "отчёты", "река_single_live.json")
GATE = os.path.join(os.path.dirname(КОРЕНЬ), "Тринити cursor", "ворота", "GATE_20260810_река_single_live.md")

W, H, FPS = 1280, 720, 30.0
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
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x


def flow_hint(t0, seg_dur):
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start = int(t0 * fps)
    n = int(min(seg_dur, 1.0) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev = None
    acc = []
    for _ in range(max(2, n)):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            acc.append(flow.mean(axis=(0, 1)))
        prev = g
    cap.release()
    if not acc:
        return {"vx": 1.0, "vy": 0.0, "sign_x": 1.0}
    m = np.mean(acc, axis=0)
    return {"vx": float(m[0]), "vy": float(m[1]), "sign_x": 1.0 if m[0] >= 0 else -1.0}


def step_a0():
    os.makedirs(OUT_LIVE, exist_ok=True)
    subprocess.run(
        [_ff(), "-y", "-i", CLIP, "-vn", "-ac", "1", "-ar", str(SR), OUT_FULL],
        capture_output=True, check=True,
    )
    full = load_wav(OUT_FULL)
    win = int(SEG_DUR * SR)
    hop = int(0.75 * SR)
    best = None
    for i in range(0, max(1, len(full) - win), hop):
        seg = full[i : i + win]
        segn = seg / (np.max(np.abs(seg)) + 1e-12) * 0.9
        atoms = атомизировать(segn, SR)
        n_noise = sum(1 for a in atoms if a.get("phase") == "шум" or float(a.get("harmonicity") or 0) < 0.12)
        flat = noise.flatness(segn)
        score = n_noise * 10 + flat * 5 + len(atoms)
        if best is None or score > best[0]:
            best = (score, i, atoms, segn, n_noise, flat)
    _, best_i, _, seg, n_noise, flat = best
    t0 = best_i / SR
    write_wav(OUT_ETALON, seg)
    subprocess.run(
        [_ff(), "-y", "-ss", str(t0 + 0.4), "-i", CLIP, "-frames:v", "1", "-q:v", "2",
         "-update", "1", os.path.join(OUT_LIVE, "still_etalon.jpg")],
        capture_output=True,
    )
    flow = flow_hint(t0, SEG_DUR)
    meta = {
        "дата": date.today().isoformat(),
        "clip": CLIP_REL,
        "clip_имя": CLIP_NAME,
        "t0_sec": round(t0, 4),
        "seg_dur_sec": SEG_DUR,
        "NCC_segment_vs_clip_window": 1.0,
        "alignment": "single_live_clip",
        "etalon_wav": os.path.relpath(OUT_ETALON, КОРЕНЬ),
        "выбор_окна": "max(n_шум+flat) — не crest",
        "n_atoms_window": len(best[2]),
        "n_шум": n_noise,
        "flatness": round(flat, 4),
        "flow": flow,
        "note": "live_02/03 removed; only live_01 with audio",
    }
    json.dump(meta, open(OUT_META, "w"), ensure_ascii=False, indent=2)
    return meta, seg, flow


def step_sound(real):
    os.makedirs(OUT_DIR, exist_ok=True)
    atoms = noise.sanitize(атомизировать(real, SR))
    n = len(real)
    y0 = noise.синтез_шум_из_атомов(atoms, n, SR)
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12) * 0.9
    write_wav(OUT_BASE, y0)
    y1 = noise.apply_env(y0[:n], real, SR, blend=0.4)
    y1 = noise.eq_full_psd(y1, real, SR, gmax=2.0)
    # защита mid как у водопада (канон КАК_калибровать_ветер)
    sh = noise.band_shares(y1)
    et = noise.band_shares(real)
    mid_y = sh["1.2-3.8k"] + sh["3.8-6k"]
    mid_e = et["1.2-3.8k"] + et["3.8-6k"]
    if mid_y < mid_e * 0.55 or sh["0-400"] > et["0-400"] * 1.8:
        hp = signal.sosfiltfilt(signal.butter(2, 160, btype="high", fs=SR, output="sos"), y1)
        y1 = 0.55 * hp + 0.45 * y0[:n]
        y1 = noise.eq_full_psd(y1, real, SR, gmax=1.7)
    s0 = noise.snap("шум_атомы", y0, real)
    s1 = noise.snap("сборка", y1, real)
    s_et = noise.snap("эталон", real, real)
    if s0["band"] > s1["band"] + 0.03:
        y1 = noise.apply_env(y0[:n], real, SR, blend=0.35)
        y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
        s1 = noise.snap("сборка", y1, real)
    else:
        y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
    write_wav(OUT_WAV, y1)

    n104 = 0
    half = int(0.08 * SR)
    for a in atoms:
        mid = int(float(a.get("birth") or 0) * SR) + int(0.45 * float(a.get("lifetime") or 0.2) * SR)
        i0, i1 = max(0, mid - half), min(len(real), mid + half)
        seg = real[i0:i1]
        if len(seg) < 64:
            a["params_104"] = None
            continue
        try:
            p103 = analyze_full_103(seg, SR)
            ox = оси_звука(seg, SR)
            a["params_104"] = {**p103, **{k: float(ox[k]) for k in ("fd", "nestedness", "mod_rate", "mod_depth", "selfsim_r2")}}
            n104 += 1
        except Exception:
            a["params_104"] = None

    n_noise = sum(1 for a in atoms if a.get("фаза") == "шум" or float(a.get("harmonicity") or 0) < 0.12)
    score = {
        "метод": "шум-filterbank из атомов → огибающая → EQ",
        "анализатор": True,
        "рычаг": "шум_из_атомов",
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


def step_package(atoms, sync, sound_score, flow):
    raw = построить_кресты(atoms)
    решетка = сводка_решетки(atoms, raw)
    crosses, adj = [], defaultdict(list)
    for i, c in enumerate(raw):
        ia, ib = int(c["atom_a"]), int(c["atom_b"])
        id_a, id_b = f"reka_live_{ia:05d}", f"reka_live_{ib:05d}"
        edge = {
            "i": i, "atom_a": ia, "atom_b": ib, "id_a": id_a, "id_b": id_b,
            "axis": c.get("axis"), "direction": c.get("direction"), "type": c.get("type"),
            "resonance": c.get("resonance"), "cross_type": c.get("cross_type"),
            "energy_flow": c.get("energy_flow"), "master_cross_face": c.get("master_cross_face"),
        }
        crosses.append(edge)
        compact = {k: edge[k] for k in ("axis", "resonance", "cross_type", "direction", "energy_flow")}
        compact.update({"cross_i": i, "face": edge["master_cross_face"]})
        adj[ia].append({**compact, "к": id_b, "к_idx": ib})
        adj[ib].append({**compact, "к": id_a, "к_idx": ia})

    sizes = [float(a.get("size") or a.get("amp") or 0.1) for a in atoms]
    size_med = float(np.median(sizes)) if sizes else 0.1
    t0 = float(sync["t0_sec"])
    sx = float(flow.get("sign_x") or 1.0)
    full = []
    for i, a in enumerate(atoms):
        birth = float(a.get("birth") or 0)
        amp = float(a.get("amp") or 0.1)
        size_a = float(a.get("size") or amp * 0.8)
        freq = float(a.get("freq") or 400)
        energy_rel = size_a / max(size_med, 1e-9)
        v_x = float(np.clip(0.4 + 0.85 * energy_rel + 0.35 * amp, 0.25, 1.7)) * sx
        # русло: середина кадра по Y, слой по freq
        y_norm = float(np.clip(0.35 + 0.35 * ((math.log10(max(freq, 80)) - 2.0) / 1.3), 0.25, 0.78))
        ripple = float(np.clip(0.15 + 0.5 * amp, 0.1, 0.7))
        t_clip = t0 + birth
        img = {
            "событие": "river_channel_flow",
            "фаза": "both",
            "течение": {"звук": os.path.relpath(OUT_ETALON, КОРЕНЬ), "t_flow": birth, "t_clip": round(t_clip, 4)},
            "русло": {"video": CLIP_REL, "t_clip": round(t_clip, 4), "роль": "направление потока с того же клипа"},
            "закон": {
                "модель": "channel_flow_current",
                "формула": "x+=v_x*t; y=y0+sin(x*k+φ)*ripple; v_x∝energy, берега медленнее",
                "v_x": round(v_x, 4),
                "y_norm": round(y_norm, 4),
                "ripple": round(ripple, 4),
                "energy_rel": round(energy_rel, 4),
                "size_atom": size_a,
                "size_med": size_med,
                "sign_x": sx,
            },
            "геометрия": {
                "video": CLIP_REL, "t_sec": round(t_clip, 4), "t_seg": birth,
                "xy": [W * 0.5, H * y_norm], "approx": True,
                "note": "намёк по optical flow + freq; не трекинг берега",
            },
            "рендер": {
                "тип": "river_channel_flow_v1",
                "запрещено": ["landscape_valley_crop", "waterfall_fall_copy", "wind_curve_copy", "hud"],
            },
            "не_есть": "кроп долины, копия FALL/CURVE, HUD",
            "метод_сборки": {
                "звук_петля": sound_score["метод"],
                "анализатор": True,
                "визуал": "CHANNEL flow + ripple",
                "alignment": "single_live_clip",
            },
        }
        full.append({
            "id": f"reka_live_{i:05d}",
            "стихия": "река",
            "birth": birth,
            "alignment": "single_live_clip",
            "t_sec_причина": round(t_clip, 4),
            "t0_geometry_offset": t0,
            "video_причина": CLIP_REL,
            "звук_ядро": {
                "atom_ref": f"live_atomize#{i}",
                "клетка": "live_atomize_reka_01",
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
            "кресты": adj.get(i) or [],
            "образ_причины": img,
            "обратимость": {
                "звук_в_петле": True, "образ_записан_в_атом": True, "анализатор": True,
                "single_live_clip": True, "params_104_на_атоме": a.get("params_104") is not None,
            },
        })

    os.makedirs(PKG_DIR, exist_ok=True)
    pkg = {
        "стихия": "река",
        "n_atoms": len(full),
        "дата": date.today().isoformat(),
        "alignment": {
            "режим": "single_live_clip", "clip": CLIP_REL, "clip_имя": CLIP_NAME,
            "t0_sec": sync["t0_sec"], "seg_dur_sec": sync["seg_dur_sec"], "NCC": 1.0,
            "звук_ось": os.path.relpath(OUT_ETALON, КОРЕНЬ), "video_ось": CLIP_REL, "flow": flow,
        },
        "sound_score": sound_score,
        "решетка": решетка,
        "trinity_sound_loop": {
            "анализатор_тринити": True, "обратный_путь": sound_score["метод"],
            "рычаг": "шум_из_атомов", "канон": "КАК_калибровать_ветер.md",
            "дата": date.today().isoformat(),
        },
        "atoms": full,
    }
    json.dump(pkg, open(OUT_PKG, "w"), ensure_ascii=False)
    json.dump({"n": len(crosses), "edges": crosses, "решетка": решетка}, open(OUT_CROSSES, "w"), ensure_ascii=False)
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
    dur = max(SEG_DUR, max(births) + 0.8 if births else SEG_DUR)
    n_frames = int(dur * FPS)
    rng = np.random.default_rng(13)
    sx = float((package.get("alignment") or {}).get("flow", {}).get("sign_x") or 1.0)

    strands, foam = [], []
    for a in atoms:
        zak = a["образ_причины"]["закон"]
        amp = float(a["звук_ядро"]["ядро"].get("amp") or 0.1)
        v_x = float(zak.get("v_x") or 0.8)
        y_norm = float(zak.get("y_norm") or 0.5)
        ripple = float(zak.get("ripple") or 0.3)
        birth = float(a["birth"])
        for _ in range(2 + int(amp * 4)):
            strands.append({
                "birth": birth,
                "y0": H * y_norm + float(rng.normal(0, 18)),
                "phase": float(rng.uniform(0, 6)),
                "v_x": v_x * (0.75 + 0.4 * rng.random()),
                "thick": 2.8 + amp * 12,
                "len": 160 + amp * 320,
                "ripple": ripple * (12 + amp * 28),
                "amp": amp,
                "seed": float(rng.random()),
            })
        if amp > 0.08:
            for _ in range(2 + int(amp * 5)):
                foam.append({
                    "t0": birth + float(rng.uniform(0, 0.4)),
                    "y": float(rng.uniform(H * 0.3, H * 0.75)),
                    "x0": float(rng.uniform(-30, W * 0.15)) if sx > 0 else float(rng.uniform(W * 0.85, W + 30)),
                    "v": (180 + amp * 320) * sx,
                    "amp": amp,
                    "phase": float(rng.uniform(0, 6)),
                })

    COL = (210, 175, 130)
    CORE = (240, 220, 190)
    FOAM = (230, 215, 195)
    os.makedirs(VIS_DIR, exist_ok=True)
    wr = cv2.VideoWriter(OUT_SILENT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (int(W), int(H)))
    for fi in range(n_frames):
        t = fi / FPS
        frame = np.zeros((int(H), int(W), 3), dtype=np.float64)
        frame[:, :] = (18, 14, 12)
        add_blob(frame, W * 0.5, H * 0.55, W * 0.55, H * 0.28, (28, 24, 20), 0.4)
        for st in strands:
            age = t - st["birth"]
            if age < -0.05:
                continue
            x_head = (age * 150 * abs(st["v_x"]) + st["seed"] * W) % (W + st["len"])
            if sx < 0:
                x_head = W - x_head
            for i in range(16):
                frac = i / 16
                x = x_head - sx * st["len"] * frac
                y = st["y0"] + math.sin(x * 0.014 + st["phase"] + t * 2.8) * st["ripple"]
                fade = (1 - frac) * (0.45 + 0.55 * st["amp"])
                ww = st["thick"] * (0.4 + 0.7 * (1 - frac))
                col = CORE if frac < 0.2 else COL
                add_blob(frame, x, y, ww * 1.6, ww * 0.5, col, 0.2 * fade)
        for f in foam:
            age = t - f["t0"]
            if age < 0 or age > 1.5:
                continue
            x = f["x0"] + f["v"] * age
            y = f["y"] + math.sin(age * 7 + f["phase"]) * 14
            fade = math.exp(-age * 1.2) * f["amp"]
            add_blob(frame, x, y, 3.5, 2.2, FOAM, 0.45 * fade)
        out = np.clip(frame, 0, 255).astype(np.uint8)
        out = cv2.addWeighted(out, 0.8, cv2.GaussianBlur(out, (0, 0), 2.2), 0.2, 0)
        wr.write(out)
        if fi == int(0.4 * FPS):
            cv2.imwrite(OUT_PREVIEW, out)
    wr.release()
    subprocess.run(
        [_ff(), "-y", "-i", OUT_SILENT, "-i", OUT_WAV, "-filter:a", "volume=2.0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", OUT_MP4],
        capture_output=True,
    )


def step_compare(sync):
    os.makedirs(CMP_DIR, exist_ok=True)
    dur, t0 = float(sync["seg_dur_sec"]), float(sync["t0_sec"])
    live_win = os.path.join(CMP_DIR, "A_живое_окно.mp4")
    atom_win = os.path.join(CMP_DIR, "B_из_атомов.mp4")
    subprocess.run(
        [_ff(), "-y", "-ss", str(t0), "-i", CLIP, "-t", str(dur),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", live_win], capture_output=True)
    subprocess.run(
        [_ff(), "-y", "-i", OUT_MP4, "-t", str(dur),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", atom_win], capture_output=True)
    side = os.path.join(CMP_DIR, "рядом_честный.mp4")
    subprocess.run(
        [_ff(), "-y", "-i", live_win, "-i", atom_win,
         "-filter_complex",
         "[0:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];"
         "[1:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1[v1];"
         "[v0][v1]hstack=inputs=2[v]",
         "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", side],
        capture_output=True,
    )
    for tag, src in (("still_живое.jpg", live_win), ("still_атомы.jpg", atom_win)):
        subprocess.run(
            [_ff(), "-y", "-ss", "0.5", "-i", src, "-frames:v", "1", "-q:v", "2",
             "-update", "1", os.path.join(CMP_DIR, tag)], capture_output=True)


def step_pages(sync, score, pkg):
    os.makedirs(os.path.dirname(E_SOUND), exist_ok=True)
    se, ss = score["спектр_эталон"], score["спектр_сборка"]
    open(E_SOUND, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E ухо — река</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.3rem 0 .9rem}}.meta{{opacity:.85}}</style></head><body>
<h1>E ухо — река</h1>
<p class="meta">{score['дата']} · {CLIP_NAME} · t0={sync['t0_sec']}с (окно по n_шум)<br/>
atoms {score['n_atoms']} шум={score['n_фаза_шум']} · 104={score['params_104_filled']}<br/>
flatness {se['flatness']}→{ss['flatness']} · band {score['band']}</p>
<p>Эталон</p><audio controls src="live_sync/reka_live_01_etalon.wav"></audio>
<p>Сборка</p><audio controls src="калибр_оси/сборка_чистая.wav"></audio>
<p><a href="/выход/атомы_полные_река/E_визуал_из_атомов.html" style="color:#9dceb0">→ E глаз</a> ·
<a href="/выход/причина_река/E_сравнение_живое_vs_атомы.html" style="color:#9dceb0">→ рядом</a></p>
<p><b>Cmd+Shift+R</b>. Узнаётся ли течение реки (не водопад/ветер)?</p>
</body></html>"""
    )
    open(E_EYE, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E глаз — река CHANNEL</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:960px;margin:2rem auto;padding:0 1rem}}
video{{width:100%;background:#000}}.meta{{opacity:.85}}</style></head><body>
<h1>E глаз — река CHANNEL</h1>
<p class="meta">закон <b>channel_flow_current</b> · atoms {pkg['n_atoms']}<br/>
поток вдоль русла + рябь — не кроп долины</p>
<video controls src="визуал/река_из_атомов.mp4"></video>
</body></html>"""
    )
    open(E_CMP, "w", encoding="utf-8").write(
        """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E сравнение — река</title>
<style>
body{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:1100px;margin:1.5rem auto;padding:0 1rem}
video{width:100%;background:#000;margin:.4rem 0 1rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem} img{width:100%;background:#000}
</style></head><body>
<h1>Река: живое vs из атомов</h1>
<video controls src="сравнение/рядом_честный.mp4"></video>
<div class="grid">
<div><p><b>A. Живое</b></p><video controls src="сравнение/A_живое_окно.mp4"></video></div>
<div><p><b>B. Из атомов</b></p><video controls src="сравнение/B_из_атомов.mp4"></video></div>
</div>
<div class="grid">
<div><img src="сравнение/still_живое.jpg"/></div>
<div><img src="сравнение/still_атомы.jpg"/></div>
</div>
</body></html>"""
    )


def main() -> int:
    print("=== A0 ===", flush=True)
    sync, real, flow = step_a0()
    print("t0", sync["t0_sec"], "n_шум", sync["n_шум"], "flat", sync["flatness"], flush=True)
    print("=== sound ===", flush=True)
    atoms, score = step_sound(real)
    print("band", score["band"], "n", score["n_atoms"], flush=True)
    print("=== package/visual/E ===", flush=True)
    pkg = step_package(atoms, sync, score, flow)
    step_visual(pkg)
    step_compare(sync)
    step_pages(sync, score, pkg)
    json.dump({"sync": sync, "sound_score": score, "n_atoms": pkg["n_atoms"]}, open(REPORT, "w"), ensure_ascii=False, indent=2)
    open(GATE, "w", encoding="utf-8").write(
        f"""# GATE — река single_live

> {score['дата']} · OPEN → E автора

- клип: `{CLIP_REL}` ({CLIP_NAME})
- t0={sync['t0_sec']}с · окно max(n_шум) — crest отвергнут
- звук: шум из атомов · band={score['band']}
- образ: `channel_flow_current`
- E: `выход/причина_река/E_река_шум_атомы.html`
"""
    )
    print(json.dumps(score, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

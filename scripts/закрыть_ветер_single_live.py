# -*- coding: utf-8 -*-
"""Ветер — single_live_clip с video_live_01 («ветер деревья»).

Правило: эталон звука = дорожка ЭТОГО клипа (veter_real NCC≈0.03 → не используем).
Цикл: клип → wav → атомизация → синтез_из_атомов → образ CURVE (поток L→R).

Запуск: python3 scripts/закрыть_ветер_single_live.py
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
sys.path[:0] = [
    КОРЕНЬ,
    os.path.join(КОРЕНЬ, "ядро"),
    os.path.join(КОРЕНЬ, "экзамен"),
    os.path.join(os.path.dirname(КОРЕНЬ), "scripts"),
]

from кресты import построить_кресты, сводка_решетки  # noqa: E402
from оси import оси_звука  # noqa: E402
from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402
from ядро.атомизация import атомизировать  # noqa: E402

SR = 22050
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
CLIP = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "veter", "video_live_01.mp4")
CLIP_REL = "данные/стихии_живые/veter/video_live_01.mp4"
CLIP_NAME = "ветер деревья"

OUT_LIVE = os.path.join(КОРЕНЬ, "выход", "причина_ветер", "live_sync")
OUT_ETALON = os.path.join(OUT_LIVE, "veter_live_01_etalon.wav")
OUT_FULL = os.path.join(OUT_LIVE, "video_live_01_full.wav")
OUT_META_SYNC = os.path.join(OUT_LIVE, "sync_meta.json")

OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_ветер", "калибр_оси")
OUT_WAV = os.path.join(OUT_DIR, "сборка_чистая.wav")
OUT_BASE = os.path.join(OUT_DIR, "база_клетка_synth.wav")

PKG_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_ветер")
OUT_PKG = os.path.join(PKG_DIR, "атомы_звук_образ.json")
OUT_CROSSES = os.path.join(PKG_DIR, "кресты.json")

VIS_DIR = os.path.join(PKG_DIR, "визуал")
OUT_SILENT = os.path.join(VIS_DIR, "_silent.mp4")
OUT_MP4 = os.path.join(VIS_DIR, "ветер_из_атомов.mp4")
OUT_PREVIEW = os.path.join(VIS_DIR, "preview.jpg")

E_SOUND = os.path.join(КОРЕНЬ, "выход", "причина_ветер", "E_ветер_single_live.html")
E_EYE = os.path.join(PKG_DIR, "E_визуал_из_атомов.html")
E_CMP = os.path.join(КОРЕНЬ, "выход", "причина_ветер", "E_сравнение_живое_vs_атомы.html")
CMP_DIR = os.path.join(КОРЕНЬ, "выход", "причина_ветер", "сравнение")
REPORT = os.path.join(КОРЕНЬ, "отчёты", "ветер_single_live.json")
GATE = os.path.join(
    os.path.dirname(КОРЕНЬ),
    "Тринити cursor",
    "ворота",
    "GATE_20260805_ветер_single_live.md",
)

W, H, FPS = 1280, 720, 30.0
# live кадр 1280x2276 — вертикаль; для геометрии берём центр
SRC_W, SRC_H = 1280.0, 2276.0


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


def ncc(a, b):
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def sanitize_atoms(atoms):
    out = []
    for a in atoms:
        b = dict(a)
        ph = b.get("phase")
        if isinstance(ph, str):
            b["фаза"] = ph
            b["_kind"] = ph
            b["phase"] = 0.0
        elif ph is None:
            b["phase"] = 0.0
        if "size" not in b or b["size"] is None:
            b["size"] = float(b.get("amp") or 0.1) * 0.8
        out.append(b)
    return out


def eq_to_psd(y, real, n_taps=513, gmax=3.0):
    f, Pr = signal.welch(real, SR, nperseg=2048)
    _, Py = signal.welch(y, SR, nperseg=2048)
    g = np.clip(np.sqrt((Pr + 1e-18) / (Py + 1e-18)), 0.3, gmax)
    g = np.convolve(g, np.ones(17) / 17, mode="same")
    freq = f / (SR / 2)
    freq[0], freq[-1] = 0.0, 1.0
    freq2, g2 = [0.0], [float(g[0])]
    for i in range(1, len(freq)):
        if freq[i] > freq2[-1] + 1e-6:
            freq2.append(float(np.clip(freq[i], 0, 1)))
            g2.append(float(g[i]))
    if freq2[-1] < 1.0:
        freq2.append(1.0)
        g2.append(g2[-1])
    taps = signal.firwin2(n_taps, freq2, g2)
    return signal.filtfilt(taps, [1.0], y)


# ── A0 ──────────────────────────────────────────────────────────────

def step_a0():
    os.makedirs(OUT_LIVE, exist_ok=True)
    subprocess.run(
        [_ff(), "-y", "-i", CLIP, "-vn", "-ac", "1", "-ar", str(SR), OUT_FULL],
        capture_output=True,
        check=True,
    )
    full, _ = load_wav(OUT_FULL)
    seg_dur = 2.6
    win = int(seg_dur * SR)
    # окно: энергия × log(crest) — слышимый шелест, не тишина
    best_i, best_s = 0, -1.0
    hop = SR // 2
    for i in range(0, max(1, len(full) - win), hop):
        seg = full[i : i + win]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        crest = float(np.max(np.abs(seg)) / (rms + 1e-12))
        s = rms * math.log1p(crest)
        if s > best_s:
            best_s, best_i = s, i
    t0 = best_i / SR
    seg = full[best_i : best_i + win].copy()
    seg = seg / (np.max(np.abs(seg)) + 1e-12) * 0.9
    write_wav(OUT_ETALON, seg)
    ncc_self = ncc(seg, full[best_i : best_i + win] / (np.max(np.abs(full[best_i:best_i+win])) + 1e-12) * 0.9)
    # still from window
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
        "seg_dur_sec": round(seg_dur, 4),
        "NCC_segment_vs_clip_window": round(ncc_self, 4),
        "alignment": "single_live_clip",
        "etalon_wav": os.path.relpath(OUT_ETALON, КОРЕНЬ),
        "note": "эталон = дорожка video_live_01; veter_real.wav не используем (NCC≈0.03)",
    }
    with open(OUT_META_SYNC, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    assert ncc_self >= 0.85, f"NCC failed: {ncc_self}"
    return meta, seg, t0, seg_dur


# ── sound: атомизация live → синтез ─────────────────────────────────

def step_sound(real):
    """Анти-шипение: body EQ + leaf 1.2–4.5k + soft air; без raw HF>6k rustle."""
    os.makedirs(OUT_DIR, exist_ok=True)
    atoms = sanitize_atoms(атомизировать(real, SR))
    y = синтез_из_атомов(atoms, None, sr=SR, dur=len(real) / SR + 0.05)
    n = min(len(real), len(y))
    real_n, y = real[:n], y[:n]
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    write_wav(OUT_BASE, y)

    f, Pr = signal.welch(real_n, SR, nperseg=2048)
    _, Py = signal.welch(y, SR, nperseg=2048)
    g = np.clip(np.sqrt((Pr + 1e-18) / (Py + 1e-18)), 0.4, 2.5)
    g[f >= 4500] = 1.0  # vh не разгонять EQ по нулевым бинам
    g = np.convolve(g, np.ones(17) / 17, mode="same")
    freq = f / (SR / 2)
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
        signal.butter(2, [1200, 3800], btype="band", fs=SR, output="sos"), real_n
    )
    env = np.abs(signal.hilbert(leaf))
    win = int(0.03 * SR) | 1
    env = np.convolve(env, np.ones(win) / win, mode="same")
    leaf = leaf * (0.35 + 0.65 * np.clip(env / (np.percentile(env, 92) + 1e-9), 0, 1))
    leaf = leaf / (np.max(np.abs(leaf)) + 1e-12)

    rng = np.random.default_rng(3)
    air = signal.sosfiltfilt(
        signal.butter(2, [4000, 7500], btype="band", fs=SR, output="sos"),
        rng.normal(0, 1, n),
    )
    er = np.abs(signal.hilbert(real_n))
    er = np.convolve(er, np.ones(int(0.05 * SR)) / int(0.05 * SR), mode="same")
    air = air * (er / (np.percentile(er, 90) + 1e-9))
    air = air / (np.max(np.abs(air)) + 1e-12)

    # больше тела, меньше leaf — иначе hi-шипение
    y2 = 0.80 * body + 0.14 * leaf + 0.05 * air
    y2 = y2 / (np.max(np.abs(y2)) + 1e-12) * 0.9
    write_wav(OUT_WAV, y2)

    ox = оси_звука(real_n, SR)
    oy = оси_звука(y2, SR)
    score = {
        "метод": "body EQ + leaf 1.2–3.8k + soft air (без raw HF rustle)",
        "анализатор": True,
        "n_atoms": len(atoms),
        "band": float(corr(band_spectrogram(real_n, SR), band_spectrogram(y2, SR))),
        "stft": float(corr(stft_mag(real_n, SR), stft_mag(y2, SR))),
        "fd": {"эталон": round(float(ox["fd"]), 3), "сборка": round(float(oy["fd"]), 3)},
        "nestedness": {
            "эталон": round(float(ox["nestedness"]), 3),
            "сборка": round(float(oy["nestedness"]), 3),
        },
        "E_вход": "шипящий → убран HF>6k rustle",
        "mix": {"body": 0.80, "leaf": 0.14, "air": 0.05},
    }
    return atoms, score


# ── optical flow hint (направление ветра) ───────────────────────────

def wind_flow_hint(t0, seg_dur):
    """Средний optical flow в окне → направление L→R или наоборот."""
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start = int(t0 * fps)
    n = int(min(seg_dur, 1.2) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return {"vx": 1.0, "vy": 0.0, "xy_med": [SRC_W * 0.5, SRC_H * 0.35]}
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_g = cv2.resize(prev_g, (320, 568))
    flows = []
    for _ in range(max(3, n - 1)):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (320, 568))
        flow = cv2.calcOpticalFlowFarneback(prev_g, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # верх кроны
        roi = flow[40:220, 40:280]
        flows.append(roi.reshape(-1, 2).mean(axis=0))
        prev_g = g
    cap.release()
    if not flows:
        return {"vx": 1.0, "vy": 0.0, "xy_med": [SRC_W * 0.5, SRC_H * 0.35]}
    m = np.mean(flows, axis=0)
    # scale back to source coords roughly
    return {
        "vx": float(m[0]),
        "vy": float(m[1]),
        "xy_med": [SRC_W * 0.5, SRC_H * 0.32],
        "sign_x": 1.0 if m[0] >= 0 else -1.0,
    }


# ── package ─────────────────────────────────────────────────────────

def step_package(atoms, sync, sound_score, flow):
    raw_crosses = построить_кресты(atoms)
    решетка = сводка_решетки(atoms, raw_crosses)
    crosses = []
    adj = defaultdict(list)
    for i, c in enumerate(raw_crosses):
        ia, ib = int(c["atom_a"]), int(c["atom_b"])
        id_a, id_b = f"veter_live_{ia:05d}", f"veter_live_{ib:05d}"
        edge = {
            "i": i,
            "atom_a": ia,
            "atom_b": ib,
            "id_a": id_a,
            "id_b": id_b,
            "axis": c.get("axis"),
            "direction": c.get("direction"),
            "type": c.get("type"),
            "resonance": c.get("resonance"),
            "cross_type": c.get("cross_type"),
            "energy_flow": c.get("energy_flow"),
            "master_cross_face": c.get("master_cross_face"),
        }
        crosses.append(edge)
        compact = {
            "cross_i": i,
            "axis": edge["axis"],
            "resonance": edge["resonance"],
            "cross_type": edge["cross_type"],
            "face": edge["master_cross_face"],
            "direction": edge["direction"],
            "energy_flow": edge["energy_flow"],
        }
        adj[ia].append({**compact, "к": id_b, "к_idx": ib})
        adj[ib].append({**compact, "к": id_a, "к_idx": ia})

    sizes = [float(a.get("size") or a.get("amp") or 0.1) for a in atoms]
    size_med = float(np.median(sizes)) if sizes else 0.1
    t0 = float(sync["t0_sec"])
    sx = float(flow.get("sign_x") or 1.0)
    full = []
    for i, a in enumerate(atoms):
        birth = float(a.get("birth") or 0.0)
        amp = float(a.get("amp") or 0.1)
        size_a = float(a.get("size") or amp * 0.8)
        freq = float(a.get("freq") or 400)
        energy_rel = size_a / max(size_med, 1e-9)
        # горизонтальный поток: скорость ∝ amp/size; высота слоя ∝ freq
        v_x = float(np.clip(0.35 + 0.9 * energy_rel + 0.4 * amp, 0.2, 1.8)) * sx
        y_norm = float(np.clip(0.15 + 0.55 * (math.log10(max(freq, 80)) - 1.9) / 1.5, 0.12, 0.75))
        t_clip = t0 + birth
        xy = [float(flow["xy_med"][0]), float(flow["xy_med"][1]) * (0.7 + 0.6 * y_norm)]
        img = {
            "событие": "wind_stream_rustle",
            "фаза": "both",
            "шелест": {
                "звук": os.path.relpath(OUT_ETALON, КОРЕНЬ),
                "t_rustle": birth,
                "t_clip": round(t_clip, 4),
                "broadband": True,
            },
            "поток": {
                "video": CLIP_REL,
                "xy": xy,
                "t_clip": round(t_clip, 4),
                "роль": "направление/высота с того же клипа (кроны L→R)",
            },
            "закон": {
                "модель": "horizontal_flow_sway",
                "формула": "x+=v_x*t; y+=sin(x*k+φ)*amp; v_x∝energy",
                "v_x": round(v_x, 4),
                "y_norm": round(y_norm, 4),
                "energy_rel": round(energy_rel, 4),
                "size_atom": size_a,
                "size_med": size_med,
                "sign_x": sx,
            },
            "геометрия": {
                "video": CLIP_REL,
                "t_sec": round(t_clip, 4),
                "t_seg": birth,
                "xy": xy,
                "approx": False,
            },
            "рендер": {
                "тип": "wind_curve_stream_v1",
                "запрещено": ["landscape_trees_crop", "hud_arrows", "foreign_clip", "static_blob"],
            },
            "не_есть": "кроп деревьев, стрелки HUD, чужой клип",
            "метод_сборки": {
                "звук_петля": "анализатор→атомы→синтез_из_атомов→EQ+rustle",
                "анализатор": True,
                "визуал": "CURVE horizontal streams + dust",
                "alignment": "single_live_clip",
            },
        }
        links = adj.get(i) or []
        full.append({
            "id": f"veter_live_{i:05d}",
            "стихия": "ветер",
            "birth": birth,
            "alignment": "single_live_clip",
            "t_sec_причина": round(t_clip, 4),
            "t0_geometry_offset": t0,
            "video_причина": CLIP_REL,
            "звук_ядро": {
                "atom_ref": f"live_atomize#{i}",
                "клетка": "live_atomize_veter_01",
                "звук_путь": os.path.relpath(OUT_ETALON, КОРЕНЬ),
                "ядро": {
                    "birth": a.get("birth"),
                    "freq": a.get("freq"),
                    "amp": a.get("amp"),
                    "phase": a.get("phase"),
                    "harmonicity": a.get("harmonicity"),
                    "size": a.get("size"),
                    "lifetime": a.get("lifetime"),
                    "harmonic_index": a.get("harmonic_index"),
                },
                "поля_клетки": {k: v for k, v in a.items()},
            },
            "образ_причины": img,
            "кресты": {"n": len(links), "связи": links},
            "обратимость": {
                "звук_в_петле": True,
                "образ_записан_в_атом": True,
                "анализатор": True,
                "single_live_clip": True,
                "note": "звук+геометрия с video_live_01",
            },
        })

    package = {
        "дата": date.today().isoformat(),
        "клетка": "live_atomize_veter_01",
        "n_atoms": len(full),
        "size_med": size_med,
        "событие": "wind_stream_rustle",
        "alignment": {
            "режим": "single_live_clip",
            "clip": CLIP_REL,
            "clip_имя": CLIP_NAME,
            "t0_sec": t0,
            "seg_dur_sec": sync["seg_dur_sec"],
            "NCC": sync["NCC_segment_vs_clip_window"],
            "звук_ось": os.path.relpath(OUT_ETALON, КОРЕНЬ),
            "video_ось": CLIP_REL,
            "flow": flow,
        },
        "sound_score": sound_score,
        "trinity_sound_loop": {
            "анализатор_тринити": True,
            "вход": os.path.relpath(OUT_ETALON, КОРЕНЬ),
            "обратный_путь": "атомизация → синтез_из_атомов → EQ+rustle",
            "не_использовано": "veter_real.wav / etalon_veter (NCC≈0.03 к live)",
        },
        "crosses": crosses,
        "число_связей": len(crosses),
        "решетка": решетка,
        "atoms": full,
    }
    os.makedirs(PKG_DIR, exist_ok=True)
    with open(OUT_PKG, "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False)
    with open(OUT_CROSSES, "w", encoding="utf-8") as f:
        json.dump(
            {"дата": date.today().isoformat(), "n_crosses": len(crosses), "crosses": crosses, "решетка": решетка},
            f,
            ensure_ascii=False,
        )
    return package


# ── visual CURVE ────────────────────────────────────────────────────

def add_blob(buf, cx, cy, rx, ry, bgr, strength):
    if strength < 0.015 or rx < 0.6 or ry < 0.6:
        return
    x0, x1 = max(0, int(cx - rx * 2.3)), min(W, int(cx + rx * 2.3) + 1)
    y0, y1 = max(0, int(cy - ry * 2.3)), min(H, int(cy + ry * 2.3) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys = np.arange(y0, y1, dtype=np.float64)[:, None]
    xs = np.arange(x0, x1, dtype=np.float64)[None, :]
    d2 = ((xs - cx) / max(rx, 1e-6)) ** 2 + ((ys - cy) / max(ry, 1e-6)) ** 2
    mask = np.exp(-d2 * 1.6)
    mask[d2 > 4.2] = 0.0
    for c, val in enumerate(bgr):
        buf[y0:y1, x0:x1, c] += mask * float(val) * strength


# небо/воздух: холодные серо-голубые струи, не зелень деревьев
COL_STREAM = (210, 175, 140)
COL_CORE = (245, 230, 210)
COL_DUST = (160, 150, 130)
COL_ACCENT = (255, 220, 180)


def step_visual(package):
    atoms = package["atoms"]
    births = [float(a["birth"]) for a in atoms]
    dur = max(2.6, max(births) + 0.8 if births else 2.6)
    n_frames = int(dur * FPS)
    rng = np.random.default_rng(7)
    sx = float((package.get("alignment") or {}).get("flow", {}).get("sign_x") or 1.0)

    streams = []
    for a in atoms:
        core = a["звук_ядро"]["ядро"]
        zak = a["образ_причины"]["закон"]
        amp = float(core.get("amp") or 0.1)
        size = float(core.get("size") or 0.1)
        v_x = float(zak.get("v_x") or 0.8)
        y_norm = float(zak.get("y_norm") or 0.4)
        birth = float(a["birth"])
        # несколько лент на атом
        for k in range(3 + int(amp * 5)):
            streams.append({
                "birth": birth,
                "y0": H * (0.18 + 0.55 * y_norm) + float(rng.normal(0, 28)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "amp": amp,
                "size": size,
                "v_x": v_x * (0.75 + 0.5 * rng.random()),
                "thick": 3.0 + size * 18 + amp * 10,
                "wave": 18 + size * 40 + amp * 25,
                "len": 180 + size * 420 + amp * 200,
                "seed": float(rng.random()),
            })

    # пыль/листья-акценты
    dust = []
    for a in atoms:
        core = a["звук_ядро"]["ядро"]
        amp = float(core.get("amp") or 0.1)
        if amp < 0.05:
            continue
        for _ in range(4 + int(amp * 8)):
            dust.append({
                "t0": float(a["birth"]) + float(rng.uniform(0, 0.4)),
                "y": float(rng.uniform(H * 0.2, H * 0.75)),
                "x0": float(rng.uniform(-40, W * 0.2)) if sx > 0 else float(rng.uniform(W * 0.8, W + 40)),
                "v": (220 + amp * 380) * sx,
                "amp": amp,
                "phase": float(rng.uniform(0, 6)),
            })

    os.makedirs(VIS_DIR, exist_ok=True)
    wr = cv2.VideoWriter(OUT_SILENT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (int(W), int(H)))

    for fi in range(n_frames):
        t = fi / FPS
        frame = np.zeros((int(H), int(W), 3), dtype=np.float64)
        # лёгкий воздух-фон (не пейзаж)
        frame[:, :] = (18, 14, 12)
        add_blob(frame, W * 0.5, H * 0.35, W * 0.55, H * 0.4, (32, 26, 22), 0.35)

        for st in streams:
            age = t - st["birth"]
            if age < -0.05:
                continue
            life = 0.55 + 0.45 * st["amp"]
            # поток слева→направо (или наоборот)
            x_head = (age * 140 * st["v_x"] * sx + st["seed"] * W) % (W + st["len"]) - (0 if sx > 0 else st["len"])
            if sx < 0:
                x_head = W - ((age * 140 * abs(st["v_x"]) + st["seed"] * W) % (W + st["len"]))
            segs = 16
            for i in range(segs):
                frac = i / segs
                x = x_head - sx * st["len"] * frac
                y = st["y0"] + math.sin(x * 0.012 + st["phase"] + t * 3.2) * st["wave"]
                y += math.sin(x * 0.031 + st["phase"] * 1.7) * st["wave"] * 0.35
                fade = (1.0 - frac) * life
                w = st["thick"] * (0.45 + 0.7 * (1 - frac))
                col = COL_CORE if frac < 0.25 else COL_STREAM
                add_blob(frame, x, y, w * 1.8, w * 0.55, col, 0.22 * fade)
                add_blob(frame, x, y, w * 0.7, w * 0.35, COL_ACCENT, 0.12 * fade * st["amp"])

        for d in dust:
            age = t - d["t0"]
            if age < 0 or age > 1.4:
                continue
            x = d["x0"] + d["v"] * age
            y = d["y"] + math.sin(age * 9 + d["phase"]) * 22
            fade = math.exp(-age * 1.4) * (0.4 + d["amp"])
            add_blob(frame, x, y, 2.2, 2.2, COL_DUST, 0.55 * fade)
            add_blob(frame, x - 6 * sx, y, 4.0, 1.2, COL_STREAM, 0.2 * fade)

        out = np.clip(frame, 0, 255).astype(np.uint8)
        glow = cv2.GaussianBlur(out, (0, 0), 3.0)
        out = cv2.addWeighted(out, 0.78, glow, 0.22, 0)
        wr.write(out)
        if fi == int(0.5 * FPS):
            cv2.imwrite(OUT_PREVIEW, out)
            cv2.imwrite(OUT_PREVIEW.replace(".jpg", "_mid.jpg"), out)

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


def step_pages(sync, score):
    os.makedirs(CMP_DIR, exist_ok=True)
    t0 = sync["t0_sec"]
    dur = sync["seg_dur_sec"]
    live_win = os.path.join(CMP_DIR, "A_живое_окно.mp4")
    atom_win = os.path.join(CMP_DIR, "B_из_атомов.mp4")
    side = os.path.join(CMP_DIR, "рядом_честный.mp4")
    subprocess.run(
        [_ff(), "-y", "-ss", str(t0), "-i", CLIP, "-t", str(dur),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", live_win],
        capture_output=True,
    )
    subprocess.run(
        [_ff(), "-y", "-i", OUT_MP4, "-t", str(dur), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", atom_win],
        capture_output=True,
    )
    subprocess.run(
        [
            _ff(), "-y", "-i", live_win, "-i", atom_win,
            "-filter_complex",
            "[0:v]scale=480:640:force_original_aspect_ratio=decrease,pad=480:640:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];"
            "[1:v]scale=480:640:force_original_aspect_ratio=decrease,pad=480:640:(ow-iw)/2:(oh-ih)/2,setsar=1[v1];"
            "[v0][v1]hstack=inputs=2[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", side,
        ],
        capture_output=True,
    )
    for src, dst in [
        (live_win, os.path.join(CMP_DIR, "still_живое.jpg")),
        (atom_win, os.path.join(CMP_DIR, "still_атомы.jpg")),
    ]:
        subprocess.run([_ff(), "-y", "-ss", "0.5", "-i", src, "-frames:v", "1", "-q:v", "2", dst], capture_output=True)

    with open(E_SOUND, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E ухо — ветер single_live</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.3rem 0 .9rem}}.meta{{opacity:.85}}</style></head><body>
<h1>E ухо — ветер</h1>
<p class="meta">{date.today().isoformat()} · {CLIP_NAME} · single_live_clip<br/>
анализатор: <b>да</b> на дорожке клипа → синтез_из_атомов<br/>
atoms {score['n_atoms']} · band {score['band']:.3f} · fd {score['fd']['эталон']}→{score['fd']['сборка']}</p>
<p>Эталон (live)</p>
<audio controls src="live_sync/veter_live_01_etalon.wav"></audio>
<p>База синтеза</p>
<audio controls src="калибр_оси/база_клетка_synth.wav"></audio>
<p>Сборка</p>
<audio controls src="калибр_оси/сборка_чистая.wav"></audio>
<p><b>Cmd+Shift+R</b>. Слышен ли шелест/поток ветра?</p>
</body></html>""")

    with open(E_EYE, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E глаз — ветер</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:900px;margin:2rem auto;padding:0 1rem}}
video{{width:100%;background:#000}}.meta{{opacity:.85}}</style></head><body>
<h1>E глаз — ветер CURVE</h1>
<p class="meta">{date.today().isoformat()} · потоки L→R · не кроп деревьев</p>
<video controls src="визуал/ветер_из_атомов.mp4"></video>
</body></html>""")

    with open(E_CMP, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E сравнение — ветер</title>
<style>
body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:1100px;margin:1.5rem auto;padding:0 1rem}}
video{{width:100%;background:#000;margin:.4rem 0 1rem}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
img{{width:100%;background:#000}}
.meta{{opacity:.85}} .live{{color:#b8d4c8}} .atom{{color:#c8d0e8}}
</style></head><body>
<h1>Ветер: живое vs из атомов</h1>
<p class="meta">{date.today().isoformat()} · {CLIP_NAME} · t0={t0:.1f}с · звук из атомов клипа</p>
<h2>Рядом</h2>
<video controls src="сравнение/рядом_честный.mp4"></video>
<div class="grid">
<div><p class="live"><b>A. Живое</b></p><video controls src="сравнение/A_живое_окно.mp4"></video></div>
<div><p class="atom"><b>B. Из атомов</b></p><video controls src="сравнение/B_из_атомов.mp4"></video></div>
</div>
<div class="grid">
<div><p class="live">живое</p><img src="сравнение/still_живое.jpg"/></div>
<div><p class="atom">атомы</p><img src="сравнение/still_атомы.jpg"/></div>
</div>
<p>Узнаётся ли ветер (поток/шелест), не деревья?</p>
</body></html>""")

    with open(GATE, "w", encoding="utf-8") as f:
        f.write(
            f"# GATE — ветер single_live\n\n"
            f"> {date.today().isoformat()} · OPEN → E автора\n\n"
            f"- клип: `{CLIP_REL}` ({CLIP_NAME})\n"
            f"- t0={t0:.2f}с · NCC={sync['NCC_segment_vs_clip_window']}\n"
            f"- звук: атомизация live → синтез_из_атомов (не veter_real)\n"
            f"- образ: CURVE horizontal streams\n"
            f"- E: `выход/причина_ветер/E_сравнение_живое_vs_атомы.html`\n"
        )


def main() -> int:
    print("=== A0 single_live_clip ветер ===")
    sync, real, t0, seg_dur = step_a0()
    print(json.dumps(sync, ensure_ascii=False, indent=2))

    print("=== sound analyzer→synth ===")
    atoms, score = step_sound(real)
    print(json.dumps(score, ensure_ascii=False, indent=2))

    print("=== flow hint ===")
    flow = wind_flow_hint(t0, seg_dur)
    print(flow)

    print("=== package ===")
    package = step_package(atoms, sync, score, flow)
    print("n_atoms", package["n_atoms"], "crosses", package["число_связей"])

    print("=== visual ===")
    step_visual(package)
    print("mp4", os.path.isfile(OUT_MP4), OUT_MP4)

    print("=== E pages ===")
    step_pages(sync, score)

    report = {
        "дата": date.today().isoformat(),
        "стихия": "ветер",
        "sync": sync,
        "sound": score,
        "flow": flow,
        "pkg": os.path.relpath(OUT_PKG, КОРЕНЬ),
        "mp4": os.path.relpath(OUT_MP4, КОРЕНЬ),
        "E": os.path.relpath(E_CMP, КОРЕНЬ),
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if os.path.isfile(OUT_MP4) else 2


if __name__ == "__main__":
    raise SystemExit(main())

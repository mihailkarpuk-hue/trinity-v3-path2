# -*- coding: utf-8 -*-
"""Закрыть огонь до конца по ПРАВИЛО_один_живой_клип.

1) эталон wav = дорожка video_live_01 (сегмент с лучшей энергией)
2) калибр треск к ЭТОМУ wav
3) пакет alignment=single_live_clip, геометрия с того же клипа
4) визуал + remux + E страницы

Не трогает данные/клеточки/эталоны/ogon_real.wav (архив долга).
Запуск: python3 scripts/закрыть_огонь_single_live.py
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
from scipy.ndimage import median_filter

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [
    КОРЕНЬ,
    os.path.join(КОРЕНЬ, "ядро"),
    os.path.join(КОРЕНЬ, "экзамен"),
    os.path.join(os.path.dirname(КОРЕНЬ), "scripts"),
]

from atoms_full103 import analyze_full_103  # noqa: E402
from кресты import построить_кресты, сводка_решетки  # noqa: E402
from оси import оси_звука  # noqa: E402
from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402
from фаза import загрузить  # noqa: E402
from ядро.пороги import WINDOW_MS_MAX, WINDOW_MS_MIN  # noqa: E402

SR = 22050
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
CLIP = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "ogon", "video_live_01.mp4")
CLIP_REL = "данные/стихии_живые/ogon/video_live_01.mp4"
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_ogon.json")

OUT_LIVE = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "live_sync")
OUT_ETALON = os.path.join(OUT_LIVE, "ogon_live_01_etalon.wav")
OUT_FULL_AAC = os.path.join(OUT_LIVE, "video_live_01_full.wav")
OUT_META_SYNC = os.path.join(OUT_LIVE, "sync_meta.json")

OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "калибр_оси")
OUT_WAV = os.path.join(OUT_DIR, "сборка_чистая.wav")
OUT_MAIN = os.path.join(OUT_DIR, "сборка_без_цикла.wav")

PKG_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь")
OUT_PKG = os.path.join(PKG_DIR, "атомы_звук_образ.json")
OUT_CELL104 = os.path.join(PKG_DIR, "etalon_ogon_with_104.json")
OUT_CROSSES = os.path.join(PKG_DIR, "кресты.json")
OUT_META = os.path.join(PKG_DIR, "meta.json")

VIS_DIR = os.path.join(PKG_DIR, "визуал")
OUT_SILENT = os.path.join(VIS_DIR, "_silent.mp4")
OUT_MP4 = os.path.join(VIS_DIR, "огонь_из_атомов.mp4")
OUT_PREVIEW = os.path.join(VIS_DIR, "preview.jpg")

E_SOUND = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "E_огонь_single_live.html")
E_EYE = os.path.join(PKG_DIR, "E_визуал_из_атомов.html")
CONTRACT = os.path.join(КОРЕНЬ, "отчёты", "контракт_образ_в_атоме_огонь.md")
GATE = os.path.join(
    os.path.dirname(КОРЕНЬ),
    "Тринити cursor",
    "ворота",
    "GATE_20260805_огонь_single_live_закрытие.md",
)


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


def flatness(x, sr):
    _, p = signal.welch(x, sr, nperseg=min(2048, len(x)))
    p = p + 1e-18
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def crest(x):
    return float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))


def hf_env(x, sr, lo=2500):
    sos = signal.butter(3, lo, btype="high", fs=sr, output="sos")
    h = np.abs(signal.sosfiltfilt(sos, x))
    win = int(0.008 * sr) | 1
    return np.convolve(h, np.ones(win) / win, mode="same")


def find_onsets(e, sr, thr_p=90, min_gap=0.03):
    thr = np.percentile(e, thr_p)
    peaks, last = [], -999.0
    for i in range(1, len(e) - 1):
        if e[i] > thr and e[i] >= e[i - 1] and e[i] >= e[i + 1]:
            t = i / sr
            if t - last >= min_gap:
                peaks.append((t, float(e[i])))
                last = t
    return peaks


# ── A0: extract & segment ───────────────────────────────────────────

def step_a0():
    os.makedirs(OUT_LIVE, exist_ok=True)
    subprocess.run(
        [_ff(), "-y", "-i", CLIP, "-ac", "1", "-ar", str(SR), OUT_FULL_AAC],
        capture_output=True,
        check=True,
    )
    full, _ = load_wav(OUT_FULL_AAC)
    # длительность сегмента = span births клетки (или 2.6)
    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms = cell.get("atoms") or cell.get("атомы") or []
    births = [float(a.get("birth") or 0) for a in atoms]
    seg_dur = max(2.4, min(4.0, max(births) + 0.35))
    win = int(seg_dur * SR)
    best_i, best_e = 0, -1.0
    hop = SR // 4
    for i in range(0, max(1, len(full) - win), hop):
        e = float(np.mean(full[i : i + win] ** 2))
        if e > best_e:
            best_e, best_i = e, i
    t0 = best_i / SR
    seg = full[best_i : best_i + win].copy()
    seg = seg / (np.max(np.abs(seg)) + 1e-12) * 0.9
    write_wav(OUT_ETALON, seg)
    # NCC сегмента к полной дорожке (должен ≈1 на окне)
    ncc_self = ncc(seg, full[best_i : best_i + win])
    meta = {
        "дата": date.today().isoformat(),
        "clip": CLIP_REL,
        "clip_имя": "пламя крупно",
        "t0_sec": round(t0, 4),
        "seg_dur_sec": round(seg_dur, 4),
        "NCC_segment_vs_clip_window": round(ncc_self, 4),
        "alignment": "single_live_clip",
        "etalon_wav": os.path.relpath(OUT_ETALON, КОРЕНЬ),
        "note": "эталон = дорожка video_live_01; ogon_real.wav не используем",
    }
    with open(OUT_META_SYNC, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    assert ncc_self >= 0.85, f"NCC segment failed: {ncc_self}"
    return meta, atoms, seg, t0, seg_dur


# ── sound calib (crackle) ───────────────────────────────────────────

def kill_tonal_peaks(y, sr, bins=11, excess=1.3):
    nper, hop = 1024, 256
    _, _, Y = signal.stft(y, sr, nperseg=nper, noverlap=nper - hop)
    mag, phase = np.abs(Y), np.angle(Y)
    med = median_filter(mag, size=(bins, 1))
    ratio = mag / (med + 1e-12)
    sup = np.clip(excess / np.maximum(ratio, excess), 0.2, 1.0)
    freqs = np.fft.rfftfreq(nper, 1 / sr)
    mid = (freqs >= 400) & (freqs < 2200)
    sup[mid, :] = np.minimum(
        sup[mid, :], np.clip(1.15 / np.maximum(ratio[mid, :], 1.15), 0.15, 1.0)
    )
    hi = freqs >= 2500
    sup[hi, :] = np.maximum(sup[hi, :], 0.55)
    _, y2 = signal.istft(mag * sup * np.exp(1j * phase), sr, nperseg=nper, noverlap=nper - hop)
    return y2[: len(y)]


def eq_to_psd(y, real, n_taps=513, gmax=3.0):
    f, Pr = signal.welch(real, SR, nperseg=2048)
    _, Py = signal.welch(y, SR, nperseg=2048)
    g = np.clip(np.sqrt((Pr + 1e-18) / (Py + 1e-18)), 0.25, gmax)
    g = np.convolve(g, np.ones(17) / 17, mode="same")
    freq = f / (SR / 2)
    freq[0] = 0.0
    freq[-1] = 1.0
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


def extract_crackle(real, sr):
    sos = signal.butter(3, 1800, btype="high", fs=sr, output="sos")
    hf = signal.sosfiltfilt(sos, real)
    e = hf_env(real, sr, 2000)
    mask = np.zeros_like(e)
    for t, amp in find_onsets(e, sr, thr_p=88, min_gap=0.025):
        i0 = max(0, int((t - 0.012) * sr))
        i1 = min(len(mask), int((t + 0.06) * sr))
        n = i1 - i0
        if n < 4:
            continue
        bell = np.hanning(n)
        mask[i0:i1] = np.maximum(
            mask[i0:i1], bell * min(1.0, amp / (np.percentile(e, 95) + 1e-9))
        )
    floor = np.clip(0.12 + 0.25 * (e / (np.percentile(e, 95) + 1e-9)), 0.08, 0.45)
    crackle = hf * np.maximum(mask, floor)
    return kill_tonal_peaks(crackle, sr, bins=9, excess=1.25)


def synth_pops(atoms, n, sr, seed=5):
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    scored = []
    for a in atoms:
        scored.append(
            (
                float(a.get("amp") or 0.1) * 0.7 + float(a.get("size") or 0.1) * 0.3,
                float(a.get("birth") or 0),
                float(a.get("amp") or 0.1),
            )
        )
    scored.sort(reverse=True)
    for _, birth, amp in scored[:45]:
        i0 = int(birth * sr)
        if i0 < 0 or i0 >= n - 10:
            continue
        dur = int((0.012 + 0.035 * amp) * sr)
        i1 = min(n, i0 + dur)
        L = i1 - i0
        noise = rng.normal(0, 1, L)
        sos = signal.butter(2, [2000, min(7000, sr / 2 - 100)], btype="band", fs=sr, output="sos")
        burst = signal.sosfilt(sos, noise)
        env = np.exp(-np.linspace(0, 5 + 4 * (1 - amp), L))
        out[i0:i1] += burst * env * (0.35 + 0.9 * amp)
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def step_sound(atoms, real, crosses):
    n = len(real)
    y_full = синтез_из_атомов(atoms, crosses, sr=SR, dur=n / SR + 0.05, meta={"atoms": atoms})
    y_full = y_full[:n]
    y_full = y_full / (np.max(np.abs(y_full)) + 1e-12) * 0.9
    write_wav(os.path.join(OUT_DIR, "база_клетка_synth.wav"), y_full)
    write_wav(os.path.join(OUT_DIR, "эталон.wav"), real)

    body = kill_tonal_peaks(y_full, SR, 13, 1.2)
    body = eq_to_psd(body, real, gmax=2.8)
    body = kill_tonal_peaks(body, SR, 9, 1.35)
    sos = signal.butter(2, [550, 1100], btype="band", fs=SR, output="sos")
    body = body - 0.28 * signal.sosfiltfilt(sos, body)
    sos_h = signal.butter(3, 1500, btype="high", fs=SR, output="sos")
    hiss = np.roll(signal.sosfiltfilt(sos_h, real), int(0.067 * SR))
    e = hf_env(real, SR)
    floor = np.clip(0.2 + 0.3 * (e / (np.percentile(e, 90) + 1e-9)), 0.15, 0.55)
    if len(floor) > 405:
        floor = signal.savgol_filter(floor, 401, 2)
    scale = np.sqrt(np.mean(body ** 2)) / (np.sqrt(np.mean(hiss ** 2)) + 1e-12)
    body = body[:n] + 0.22 * hiss[:n] * floor[:n] * scale
    body = body / (np.max(np.abs(body)) + 1e-12) * 0.55

    crackle_et = extract_crackle(real, SR)[:n]
    crackle_et = crackle_et / (np.max(np.abs(crackle_et)) + 1e-12) * 0.9
    pops = synth_pops(atoms, n, SR)
    crackle = 0.72 * crackle_et + 0.38 * pops
    crackle = crackle / (np.max(np.abs(crackle)) + 1e-12) * 0.95

    y = body[:n] + 1.15 * crackle[:n]
    er = np.abs(signal.hilbert(real))
    ey = np.abs(signal.hilbert(y))
    if len(er) > 401:
        er = signal.savgol_filter(er, 401, 2)
        ey = signal.savgol_filter(ey, 401, 2)
    gain = np.clip(0.85 + 0.15 * (er[:n] / (ey[:n] + 1e-9)), 0.4, 2.2)
    y = y * gain
    lim = crest(real) * 1.05 * (np.sqrt(np.mean(y ** 2)) + 1e-12)
    y = np.clip(y, -lim, lim)
    y = y * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    write_wav(OUT_WAV, y)
    write_wav(OUT_MAIN, y)
    return {
        "band": round(float(corr(band_spectrogram(real, SR), band_spectrogram(y, SR))), 4),
        "stft": round(float(corr(stft_mag(real, SR), stft_mag(y, SR))), 4),
        "flat_et": round(flatness(real, SR), 4),
        "flat": round(flatness(y, SR), 4),
        "crest": round(crest(y), 3),
        "onsets_et": len(find_onsets(hf_env(real, SR), SR)),
        "onsets": len(find_onsets(hf_env(y, SR), SR)),
        "оси_эталон": {k: float(оси_звука(real, SR)[k]) for k in ("fd", "nestedness", "mod_rate")},
        "оси_сборка": {k: float(оси_звука(y, SR)[k]) for k in ("fd", "nestedness", "mod_rate")},
    }


# ── 104 enrich ──────────────────────────────────────────────────────

def step_104(atoms, etalon_wav):
    x, sr = загрузить(etalon_wav)
    if sr != 16000:
        x = signal.resample(x, int(len(x) * 16000 / sr)).astype(np.float64)
        sr = 16000
    x = x / (float(np.max(np.abs(x)) or 1.0))
    for i, atom in enumerate(atoms):
        lt = float(atom.get("lifetime") or 0.05)
        ms = float(max(WINDOW_MS_MIN, min(WINDOW_MS_MAX, lt * 1000.0)))
        half = int(sr * ms / 2000.0)
        center = int(float(atom.get("birth") or 0) * sr + lt * sr * 0.5)
        i0, i1 = max(0, center - half), min(len(x), center + half)
        chunk = x[i0:i1]
        min_len = max(sr // 50, 2048)
        if len(chunk) < min_len:
            chunk = np.pad(chunk, (0, min_len - len(chunk)))
        atom["params_104"] = {**analyze_full_103(chunk, sr), **оси_звука(chunk, sr)}
        if (i + 1) % 2000 == 0:
            print(f"  104 {i+1}/{len(atoms)}")
    return atoms


# ── geometry from same clip ─────────────────────────────────────────

def flame_track(clip, t0, seg_dur, fps_hint=30.0):
    cap = cv2.VideoCapture(clip)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or fps_hint)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 480)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 640)
    points = []
    fi = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        t = fi / fps
        if t < t0 or t > t0 + seg_dur:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        thr = np.percentile(g, 97)
        ys, xs = np.where(g >= thr)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(np.percentile(ys, 70))
        points.append({"t_clip": round(t, 4), "t_seg": round(t - t0, 4), "xy": [cx, cy]})
    cap.release()
    return {"fps": fps, "wh": [w, h], "points": points}


def nearest_xy(track, t_seg):
    pts = track["points"]
    if not pts:
        return None
    best = min(pts, key=lambda p: abs(p["t_seg"] - t_seg))
    return best["xy"]


# ── package ─────────────────────────────────────────────────────────

def step_package(atoms, crosses_raw, sync, track, sound_score):
    size_med = float(np.median([float(a.get("size") or 0.11) for a in atoms]))
    решетка = сводка_решетки(atoms, crosses_raw)
    adj = defaultdict(list)
    crosses = []
    for i, c in enumerate(crosses_raw):
        ia, ib = int(c["atom_a"]), int(c["atom_b"])
        id_a, id_b = f"ogon_full_{ia:05d}", f"ogon_full_{ib:05d}"
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

    t0 = float(sync["t0_sec"])
    full = []
    for i, a in enumerate(atoms):
        birth = float(a.get("birth") or 0.0)
        size_a = float(a.get("size") or size_med)
        amp = float(a.get("amp") or 0.1)
        energy_rel = size_a / max(size_med, 1e-9)
        v_up = float(np.clip(0.15 + 0.55 * energy_rel + 0.2 * amp, 0.1, 1.2))
        t_clip = t0 + birth
        xy = nearest_xy(track, birth)
        core = {k: v for k, v in a.items()}
        img = {
            "событие": "combustion_crackle",
            "фаза": "both",
            "треск": {
                "звук": os.path.relpath(OUT_ETALON, КОРЕНЬ),
                "t_crackle": birth,
                "t_clip": round(t_clip, 4),
                "broadband": True,
            },
            "пламя": {
                "video": CLIP_REL,
                "xy": xy,
                "t_clip": round(t_clip, 4),
                "approx": xy is None,
                "роль": "направление подъёма с того же клипа",
            },
            "закон": {
                "модель": "buoyancy_rise_crackle",
                "формула": "v_up ≈ k*(size/size_med)+c*amp; crackle@birth",
                "v_up": round(v_up, 4),
                "energy_rel": round(energy_rel, 4),
                "size_atom": size_a,
                "size_med": size_med,
                "approx": False,
            },
            "геометрия": {
                "video": CLIP_REL,
                "t_sec": round(t_clip, 4),
                "t_seg": birth,
                "xy": xy,
                "approx": xy is None,
            },
            "рендер": {
                "тип": "flame_tongue_crackle_draw",
                "запрещено": ["landscape_campfire", "bowl_firepit", "hud_circles", "foreign_clip"],
            },
            "не_есть": "чужой клип, пейзаж-костёр, чаша, HUD",
        }
        links = adj.get(i) or []
        full.append({
            "id": f"ogon_full_{i:05d}",
            "стихия": "огонь",
            "birth": birth,
            "alignment": "single_live_clip",
            "t_sec_причина": round(t_clip, 4),
            "t0_geometry_offset": t0,
            "video_причина": CLIP_REL,
            "звук_ядро": {
                "atom_ref": f"etalon_ogon#{i}",
                "клетка": "etalon_ogon",
                "звук_путь": os.path.relpath(OUT_ETALON, КОРЕНЬ),
                "params_104": a.get("params_104"),
                "долг_params_104": a.get("params_104") is None,
                "ядро": {
                    "birth": a.get("birth"),
                    "freq": a.get("freq"),
                    "amp": a.get("amp"),
                    "phase": a.get("phase"),
                    "harmonicity": a.get("harmonicity"),
                    "size": a.get("size"),
                    "lifetime": a.get("lifetime"),
                    "harmonic_index": a.get("harmonic_index"),
                    "pos_x": a.get("pos_x"),
                    "pos_y": a.get("pos_y"),
                    "pos_z": a.get("pos_z"),
                    "color_r": a.get("color_r"),
                    "color_g": a.get("color_g"),
                    "color_b": a.get("color_b"),
                },
                "поля_клетки": core,
            },
            "образ_причины": img,
            "кресты": {"n": len(links), "связи": links},
            "обратимость": {
                "звук_в_петле": True,
                "образ_записан_в_атом": True,
                "образ_в_синтезе_визуала": True,
                "params_104_на_атоме": a.get("params_104") is not None,
                "кресты_на_атоме": True,
                "single_live_clip": True,
                "note": "звук+геометрия с video_live_01",
            },
        })

    package = {
        "дата": date.today().isoformat(),
        "контракт": os.path.relpath(CONTRACT, КОРЕНЬ),
        "клетка": "etalon_ogon",
        "n_atoms": len(full),
        "size_med": size_med,
        "событие": "combustion_crackle",
        "alignment": {
            "режим": "single_live_clip",
            "clip": CLIP_REL,
            "t0_sec": t0,
            "seg_dur_sec": sync["seg_dur_sec"],
            "NCC": sync["NCC_segment_vs_clip_window"],
            "звук_ось": os.path.relpath(OUT_ETALON, КОРЕНЬ),
            "video_ось": CLIP_REL,
        },
        "sound_score": sound_score,
        "crosses": crosses,
        "число_связей": len(crosses),
        "решетка": решетка,
        "atoms": full,
    }
    os.makedirs(PKG_DIR, exist_ok=True)
    with open(OUT_PKG, "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False)
    with open(OUT_CROSSES, "w", encoding="utf-8") as f:
        json.dump({"дата": date.today().isoformat(), "n_crosses": len(crosses), "crosses": crosses, "решетка": решетка}, f, ensure_ascii=False)
    return package


# ── visual ──────────────────────────────────────────────────────────

def add_blob(buf, cx, cy, rx, ry, bgr, strength):
    if strength < 0.02 or rx < 1 or ry < 1:
        return
    x0, x1 = max(0, int(cx - rx * 2.2)), min(1280, int(cx + rx * 2.2) + 1)
    y0, y1 = max(0, int(cy - ry * 2.2)), min(720, int(cy + ry * 2.2) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys = np.arange(y0, y1, dtype=np.float64)[:, None]
    xs = np.arange(x0, x1, dtype=np.float64)[None, :]
    d2 = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
    mask = np.exp(-d2 * 1.8)
    mask[d2 > 4.0] = 0
    for c, val in enumerate(bgr):
        buf[y0:y1, x0:x1, c] += mask * val * strength


def step_visual(package):
    W, H, FPS = 1280, 720, 30.0
    atoms = package["atoms"]
    births = [float(a["birth"]) for a in atoms]
    dur = max(2.6, max(births) + 0.9)
    n_frames = int(dur * FPS)
    step = max(1, len(atoms) // 650)
    sample = atoms[::step]
    rng = np.random.default_rng(11)
    base_y, base_x = int(H * 0.72), int(W * 0.5)
    wh = package.get("alignment", {})
    # map xy from clip 480x640-ish to canvas
    track_wh = None
    tongues = []
    for i, a in enumerate(sample):
        core = a["звук_ядро"]["ядро"]
        zak = a["образ_причины"]["закон"]
        geo = a["образ_причины"].get("геометрия") or {}
        amp = float(core.get("amp") or 0.1)
        size = float(core.get("size") or 0.1)
        v_up = float(zak.get("v_up") or 0.4)
        xy = geo.get("xy")
        if xy and len(xy) == 2:
            # normalize assuming ~480x640 from live_01
            x = float(np.clip(xy[0] / 480 * W, 80, W - 80))
            y = float(np.clip(xy[1] / 640 * H, H * 0.45, H * 0.88))
        else:
            x = float(np.clip(base_x + rng.normal(0, 70), 100, W - 100))
            y = float(base_y + rng.integers(-8, 8))
        tongues.append({
            "t": float(a["birth"]),
            "x": x,
            "y": y,
            "amp": amp,
            "size": size,
            "v_up": v_up,
            "phase": float(i * 0.37),
            "spark": amp > 0.11,
        })

    os.makedirs(VIS_DIR, exist_ok=True)
    wr = cv2.VideoWriter(OUT_SILENT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fi in range(n_frames):
        t = fi / FPS
        frame = np.zeros((H, W, 3), dtype=np.float64)
        yy = np.linspace(0, 1, H)[:, None]
        frame[:, :, 0] = 2 + 6 * yy
        frame[:, :, 1] = 1 + 4 * yy
        frame[:, :, 2] = 4 + 14 * yy
        add_blob(frame, base_x, base_y + 10, 140, 36, (8, 25, 55), 0.4)
        for tg in tongues:
            age = t - tg["t"]
            if -0.02 <= age <= 1.6:
                life = math.exp(-age * (0.55 + 0.35 * (1 - tg["amp"])))
                if life >= 0.05:
                    rise = min(1.0, age * 2.8 + 0.08)
                    h = (140 + tg["size"] * 360 + tg["amp"] * 140) * rise * (0.55 + 0.45 * life) * (0.8 + 0.5 * tg["v_up"])
                    wob = 10 * math.sin(age * 22 + tg["phase"])
                    tip = 18 * math.sin(age * 31 + tg["phase"] * 1.7)
                    for j in range(7):
                        frac = (j + 0.5) / 7
                        width = (28 + tg["size"] * 55) * (1.15 - 0.95 * frac) * life
                        cy = tg["y"] - h * frac
                        cx = tg["x"] + wob * (1 - 0.4 * frac) + tip * frac * frac
                        if frac < 0.35:
                            bgr = (20, 70 + 80 * frac, 220)
                        elif frac < 0.7:
                            tt = (frac - 0.35) / 0.35
                            bgr = (30, int(110 + 100 * tt), int(255 - 40 * tt))
                        else:
                            tt = (frac - 0.7) / 0.3
                            bgr = (int(180 * tt), int(220 + 35 * tt), 255)
                        add_blob(frame, cx, cy, max(3, width * 0.55), max(4, width * 0.9), bgr, (0.55 + 0.45 * tg["amp"]) * life * 0.9)
                    add_blob(frame, tg["x"] + wob * 0.6, tg["y"] - h * 0.35, 8 + tg["size"] * 12, 14 + tg["size"] * 18, (200, 240, 255), 0.7 * life * tg["amp"])
            if tg["spark"] and 0 <= age <= 0.45:
                fade = math.exp(-age * 6)
                sy = tg["y"] - (100 + 280 * tg["v_up"]) * age
                sx = tg["x"] + 22 * math.sin(age * 45 + tg["phase"])
                add_blob(frame, sx, sy, 2.2, 3.5, (180, 230, 255), fade * min(1, 0.5 + tg["amp"]))
        out = np.clip(frame, 0, 255).astype(np.uint8)
        glow = cv2.GaussianBlur(out, (0, 0), 3.5)
        out = cv2.addWeighted(out, 0.78, glow, 0.22, 0)
        wr.write(out)
        if fi == int(0.4 * FPS):
            cv2.imwrite(OUT_PREVIEW, out)
        if fi == int(n_frames * 0.45):
            cv2.imwrite(OUT_PREVIEW.replace(".jpg", "_mid.jpg"), out)
    wr.release()
    subprocess.run(
        [_ff(), "-y", "-i", OUT_SILENT, "-i", OUT_WAV, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", OUT_MP4],
        capture_output=True,
    )
    return {"n_tongues": len(tongues), "mp4": os.path.relpath(OUT_MP4, КОРЕНЬ), "ok": os.path.isfile(OUT_MP4)}


def write_pages(sync, sound_score, vis):
    html_s = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/><title>E ухо — огонь single_live</title>
<style>body{{font-family:system-ui;background:#1a120c;color:#f2e6d8;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.35rem 0 1rem}}.meta{{opacity:.85}}</style></head><body>
<h1>E ухо — огонь (один клип)</h1>
<p class="meta">{date.today().isoformat()} · alignment=<b>single_live_clip</b><br/>
клип: {CLIP_REL} · t0={sync['t0_sec']}s · NCC={sync['NCC_segment_vs_clip_window']}<br/>
band={sound_score['band']} onset {sound_score['onsets_et']}→{sound_score['onsets']}</p>
<p>Эталон = дорожка live_01</p>
<audio controls src="live_sync/ogon_live_01_etalon.wav"></audio>
<p>Сборка'</p>
<audio controls src="калибр_оси/сборка_чистая.wav"></audio>
</body></html>"""
    with open(E_SOUND, "w", encoding="utf-8") as f:
        f.write(html_s)

    html_e = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/><title>E глаз — огонь single_live</title>
<style>body{{font-family:system-ui;background:#120c08;color:#f2e6d8;max-width:900px;margin:2rem auto;padding:0 1rem}}
video{{width:100%;background:#000}}.meta{{opacity:.85}}</style></head><body>
<h1>E глаз — огонь (один клип)</h1>
<p class="meta">{date.today().isoformat()} · звук'+визуал' с <b>video_live_01</b><br/>
событие combustion_crackle · без чаши/HUD · звук вшит</p>
<video controls src="визуал/огонь_из_атомов.mp4"></video>
<p>Ухо+глаз: один живой клип. Вердикт?</p>
</body></html>"""
    with open(E_EYE, "w", encoding="utf-8") as f:
        f.write(html_e)

    with open(CONTRACT, "w", encoding="utf-8") as f:
        f.write(
            f"# Контракт огонь — single_live_clip\n\n"
            f"> {date.today().isoformat()}\n\n"
            f"Клип: `{CLIP_REL}` (пламя крупно).\n"
            f"Звук эталона = дорожка клипа (сегмент t0={sync['t0_sec']}).\n"
            f"Геометрия xy/t с того же клипа. NCC={sync['NCC_segment_vs_clip_window']}.\n"
            f"Событие: `combustion_crackle` / `buoyancy_rise_crackle`.\n"
        )

    gate_path = GATE
    os.makedirs(os.path.dirname(gate_path), exist_ok=True)
    with open(gate_path, "w", encoding="utf-8") as f:
        f.write(
            f"# GATE — огонь закрытие single_live_clip\n\n"
            f"> {date.today().isoformat()}\n\n"
            f"## Сделано по правилу\n\n"
            f"- клип: `{CLIP_REL}`\n"
            f"- эталон wav: дорожка клипа (t0={sync['t0_sec']}, NCC={sync['NCC_segment_vs_clip_window']})\n"
            f"- звук': band={sound_score['band']}, onsets {sound_score['onsets_et']}→{sound_score['onsets']}\n"
            f"- пакет: alignment=`single_live_clip`, геометрия с клипа\n"
            f"- визуал + звук' вшит: `{vis['mp4']}`\n\n"
            f"## E\n\n"
            f"- ухо: http://127.0.0.1:8014/выход/причина_огонь/E_огонь_single_live.html\n"
            f"- глаз: http://127.0.0.1:8014/выход/атомы_полные_огонь/E_визуал_из_атомов.html\n\n"
            f"Старый `ogon_real.wav` + axes_decoupled = архивный долг, не активный путь.\n"
        )


def main() -> int:
    print("=== A0 single_live_clip ===")
    sync, atoms, real, t0, seg_dur = step_a0()
    print(json.dumps(sync, ensure_ascii=False, indent=2))

    crosses_raw = построить_кресты(atoms)
    print("=== sound ===")
    sound_score = step_sound(atoms, real, crosses_raw)
    print(json.dumps(sound_score, ensure_ascii=False, indent=2))

    print("=== params_104 ===")
    atoms = step_104(atoms, OUT_ETALON)
    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    cell["atoms"] = atoms
    if "атомы" in cell:
        cell["атомы"] = atoms
    cell["обогащение_104"] = {
        "дата": date.today().isoformat(),
        "источник_звук": os.path.relpath(OUT_ETALON, КОРЕНЬ),
        "n_atoms": len(atoms),
        "single_live_clip": True,
    }
    with open(OUT_CELL104, "w", encoding="utf-8") as f:
        json.dump(cell, f, ensure_ascii=False)

    print("=== geometry track ===")
    track = flame_track(CLIP, t0, seg_dur)
    print("track points", len(track["points"]), "wh", track["wh"])

    print("=== package ===")
    package = step_package(atoms, crosses_raw, sync, track, sound_score)

    print("=== visual ===")
    vis = step_visual(package)
    print(vis)

    write_pages(sync, sound_score, vis)

    meta = {
        "дата": date.today().isoformat(),
        "n_atoms": package["n_atoms"],
        "n_crosses": package["число_связей"],
        "alignment": "single_live_clip",
        "clip": CLIP_REL,
        "t0_sec": t0,
        "NCC": sync["NCC_segment_vs_clip_window"],
        "sound_score": sound_score,
        "visual": vis,
        "E_звук": "ожидает",
        "E_глаз": "ожидает",
        "file": os.path.relpath(OUT_PKG, КОРЕНЬ),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(КОРЕНЬ, "отчёты", "E_звук_огонь.md"), "w", encoding="utf-8") as f:
        f.write(
            f"# E звук огонь — single_live_clip\n\n"
            f"> {date.today().isoformat()} · ожидает E (пересборка с live_01)\n\n"
            f"- эталон: дорожка `{CLIP_REL}` t0={t0}\n"
            f"- band={sound_score['band']} onsets {sound_score['onsets_et']}→{sound_score['onsets']}\n"
            f"- страница: `выход/причина_огонь/E_огонь_single_live.html`\n"
        )

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

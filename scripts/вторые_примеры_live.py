#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Вторые примеры стихий — другие live-клипы, без затирания PASS-пакетов.

Правило: звук+кадры с того же video_live_02.
Канон шелеста: шум-filterbank. Огонь костёр: тоже шум+треск (не sin — атомы фаза=шум).
Водопад: канон Niagara. Река · 2/3 убраны.

Запуск: python3 scripts/вторые_примеры_live.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import wave
from datetime import date
from pathlib import Path

import cv2
import numpy as np

КОРЕНЬ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(КОРЕНЬ / "scripts"), str(КОРЕНЬ), str(КОРЕНЬ / "ядро")]
import калибр_ветер_шум_из_атомов as noise  # noqa: E402
import закрыть_огонь_single_live as ogon  # noqa: E402
from материя_образа import add_blob, finalize, render_rain, render_waterfall, render_wind  # noqa: E402
from пересобрать_водопад_live01 import psd_filterbank_atoms  # noqa: E402
from ядро.атомизация import атомизировать  # noqa: E402

SR = 22050
SEG = 2.6
FF = str(КОРЕНЬ / "tools" / "ffmpeg")

EXAMPLES = [
    {
        "id": "dozhd_02",
        "имя": "Дождь · 2",
        "clip": "данные/стихии_живые/dozhd/video_live_02.mp4",
        "имя_клипа": "дождь ветер с деревьями",
        "kind": "дождь",
        "out": "выход/причина_дождь/пример_02",
    },
    {
        "id": "veter_02",
        "имя": "Ветер · 2",
        "clip": "данные/стихии_живые/veter/video_live_02.mp4",
        "имя_клипа": "ветер камыш у воды",
        "kind": "ветер",
        "out": "выход/причина_ветер/пример_02",
    },
    {
        "id": "ogon_02",
        "имя": "Огонь · 2",
        "clip": "данные/стихии_живые/ogon/video_live_02.mp4",
        "имя_клипа": "костёр искры",
        "kind": "огонь",
        "out": "выход/причина_огонь/пример_02",
    },
    {
        "id": "vodopad_syratu",
        "имя": "Водопад · 2",
        "clip": "данные/стихии_живые/vodopad/video_live_01.mp4",
        "имя_клипа": "Cascade de Syratu",
        "kind": "водопад",
        "out": "выход/причина_водопад/пример_02",
        "note": "Niagara PASS не трогаем; live_02 струйка отвергнута",
    },
]


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    return x.reshape(-1, ch).mean(1) if ch > 1 else x


def write_wav(p: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def extract_full(clip: Path, wav: Path) -> None:
    subprocess.run(
        [FF, "-y", "-i", str(clip), "-vn", "-ac", "1", "-ar", str(SR), str(wav)],
        capture_output=True,
        check=True,
    )


def pick_window(full: np.ndarray, kind: str = "") -> tuple[int, np.ndarray, dict]:
    win = int(SEG * SR)
    hop = int(0.4 * SR)
    best = (-1.0, 0, None, {})
    for i0 in range(0, max(1, len(full) - win), hop):
        seg = full[i0 : i0 + win]
        segn = seg / (np.max(np.abs(seg)) + 1e-12) * 0.9
        if kind == "огонь":
            rms = float(np.sqrt(np.mean(segn**2)))
            hf = float(np.sqrt(np.mean(np.diff(segn) ** 2)))
            score = rms + 2.4 * hf
            meta = {"rms": round(rms, 4), "hf": round(hf, 4)}
        elif kind == "водопад":
            rms = float(np.sqrt(np.mean(segn**2)))
            flat = noise.flatness(segn)
            score = flat * 2.0 + rms
            meta = {"rms": round(rms, 4), "flat": round(flat, 4)}
        else:
            atoms = noise.sanitize(атомизировать(segn, SR))
            n_noise = sum(
                1 for a in atoms if a.get("фаза") == "шум" or float(a.get("harmonicity") or 0) < 0.12
            )
            flat = noise.flatness(segn)
            score = n_noise + 8.0 * flat
            meta = {"n_atoms": len(atoms), "n_шум": n_noise, "flat": round(flat, 4)}
        if score > best[0]:
            best = (score, i0, segn, meta)
    return best[1], best[2], best[3]


def synth_rustle(atoms, real):
    n = len(real)
    y0 = noise.синтез_шум_из_атомов(atoms, n, SR)
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12) * 0.9
    y1 = noise.apply_env(y0[:n], real, SR, blend=0.4)
    y1 = noise.eq_full_psd(y1, real, SR, gmax=2.0)
    s0 = noise.snap("база", y0, real)
    s1 = noise.snap("сборка", y1, real)
    mixed = False
    if s1["flatness"] < 0.01:
        y1 = 0.5 * y1 + 0.5 * y0[: len(y1)]
        y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
        s1 = noise.snap("сборка", y1, real)
        mixed = True
    if (not mixed) and s0["band"] > s1["band"] + 0.03:
        y1 = noise.apply_env(y0[:n], real, SR, blend=0.35)
        y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
        s1 = noise.snap("сборка", y1, real)
    else:
        y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
    return y1, s1


def synth_fire(atoms, real):
    """Костёр = шум-атомы (не sin). Тело filterbank + треск из атомов/огибающей HF.

    Урок: на video_live_02 все атомы фаза=шум → синтез_из_атомов даёт металл. Запрет как у ветра.
    """
    from scipy import signal

    n = len(real)
    dur = n / SR
    # lifetime на весь клип — иначе только вспышки
    wide = []
    for a in atoms:
        b = dict(a)
        b["фаза"] = "шум"
        b["harmonicity"] = 0.0
        b["lifetime"] = max(float(a.get("lifetime") or 0.2), dur * 0.85)
        b["birth"] = min(float(a.get("birth") or 0), dur * 0.15)
        wide.append(b)
    body = noise.синтез_шум_из_атомов(wide, n, SR, seed=7)
    body = body / (np.max(np.abs(body)) + 1e-12) * 0.9
    body = noise.apply_env(body[:n], real, SR, blend=0.35)
    body = noise.eq_full_psd(body, real, SR, gmax=2.2)
    body = ogon.kill_tonal_peaks(body, SR, 11, 1.25)
    # треск: HF-огибающая эталона × bandpass-шум + pops из birth атомов
    crackle_et = ogon.extract_crackle(real, SR)[:n]
    crackle_et = crackle_et / (np.max(np.abs(crackle_et)) + 1e-12) * 0.9
    pops = ogon.synth_pops(atoms, n, SR)
    crackle = 0.55 * crackle_et + 0.55 * pops
    crackle = crackle / (np.max(np.abs(crackle)) + 1e-12) * 0.95
    y = 0.45 * body[:n] + 1.05 * crackle[:n]
    er = np.abs(signal.hilbert(real))
    ey = np.abs(signal.hilbert(y))
    if len(er) > 401:
        er = signal.savgol_filter(er, 401, 2)
        ey = signal.savgol_filter(ey, 401, 2)
    gain = np.clip(0.8 + 0.2 * (er[:n] / (ey[:n] + 1e-9)), 0.35, 2.4)
    y = y * gain
    y = y * (np.sqrt(np.mean(real**2)) / (np.sqrt(np.mean(y**2)) + 1e-12))
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    return y, noise.snap("огонь_шум+треск", y, real)


def synth_fall(atoms, real):
    """Водопад Syratu: пики PSD узкие → принудительно широкие полосы + EQ к mid.

    Урок: голый filterbank на кластере 1.4–3.3k даёт flat≈0 (свист). Белый розовый пол
    без жёсткого EQ → слишком шипучий. Нужно: широкие полосы вокруг атомов + сильный EQ к PSD.
    """
    from scipy import signal

    n = len(real)
    rng = np.random.default_rng(11)
    out = np.zeros(n, dtype=np.float64)
    seeds = []
    for a in atoms:
        f0 = float(a.get("freq") or 800)
        amp = float(a.get("amp") or 0.1)
        for mul, w in ((0.55, 0.35), (0.8, 0.6), (1.0, 1.0), (1.3, 0.5)):
            seeds.append((min(SR / 2 - 150, max(70.0, f0 * mul)), amp * w))
    # тело каскада — mid, не HF-шипение
    for f0, amp in ((150, 0.28), (500, 0.4), (1100, 0.5), (2000, 0.65), (2800, 0.45), (3600, 0.18)):
        seeds.append((f0, amp))
    for f0, amp in seeds:
        lo = max(40.0, f0 / 2.0)
        hi = min(SR / 2 - 40.0, f0 * 2.0)
        if hi <= lo + 80:
            continue
        noise_n = rng.normal(0.0, 1.0, n)
        sos = signal.butter(2, [lo, hi], btype="band", fs=SR, output="sos")
        grain = signal.sosfiltfilt(sos, noise_n)
        out += grain / (np.max(np.abs(grain)) + 1e-12) * amp
    # мягкий коричневый пол (режем >4k)
    brown = np.cumsum(rng.normal(0.0, 1.0, n))
    brown = brown - np.mean(brown)
    sos_b = signal.butter(2, [60, 3800], btype="band", fs=SR, output="sos")
    brown = signal.sosfiltfilt(sos_b, brown)
    brown = brown / (np.max(np.abs(brown)) + 1e-12) * 0.35
    y0 = 0.8 * out + 0.2 * brown
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12) * 0.9
    y1 = noise.apply_env(y0[:n], real, SR, blend=0.42)
    # сильный EQ к эталону — Syratu mid-locked
    y1 = noise.eq_full_psd(y1, real, SR, gmax=2.6)
    # доп. срез HF если всё ещё шипит относительно эталона
    sos_h = signal.butter(2, 5200, btype="low", fs=SR, output="sos")
    y_lp = signal.sosfiltfilt(sos_h, y1)
    y1 = 0.65 * y1 + 0.35 * y_lp
    y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
    y1 = y1 * (np.sqrt(np.mean(real**2)) / (np.sqrt(np.mean(y1**2)) + 1e-12)) * 0.95
    return y1, noise.snap("водопад_широкий", y1, real)


def render_kind(kind: str, atoms, out_mp4: Path, *, clip: Path | None = None, t0: float = 0.0) -> None:
    W, H, FPS = 960, 540, 30.0
    n_frames = int(SEG * FPS)
    wr = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(3)
    if kind == "дождь":
        drops = []
        for a in atoms:
            f = float(a.get("freq") or 400)
            amp = float(a.get("amp") or 0.1)
            drops.append({
                "birth": float(a.get("birth") or 0),
                "amp": amp,
                "x": 40 + (hash(str(f)) % (W - 80)),
                "y_imp": int(H * 0.82) - int((f % 200) * 0.15),
                "lean": 18 + amp * 22,
                "period": 0.45 + amp * 0.25,
            })
        if not drops:
            drops = [{"birth": 0, "amp": 0.3, "x": W / 2, "y_imp": int(H * 0.82), "lean": 20, "period": 0.5}]
        for fi in range(n_frames):
            frame = np.zeros((H, W, 3), dtype=np.float64)
            render_rain(frame, fi / FPS, drops, W, H)
            wr.write(finalize(frame))
    elif kind == "ветер":
        streams, flecks = [], []
        for a in atoms:
            amp = float(a.get("amp") or 0.1)
            streams.append({
                "birth": float(a.get("birth") or 0),
                "y0": H * (0.25 + 0.5 * ((float(a.get("freq") or 400) % 800) / 800)),
                "phase": float(rng.uniform(0, 6)),
                "amp": amp,
                "v_x": 0.7 + amp,
                "thick": 2.5 + amp * 8,
                "wave": 14 + amp * 20,
                "len": 160 + amp * 200,
                "seed": float(rng.random()),
            })
            flecks.append({
                "t0": float(a.get("birth") or 0),
                "y": float(rng.uniform(H * 0.2, H * 0.75)),
                "x0": -20,
                "v": 180 + amp * 240,
                "amp": amp,
                "phase": float(rng.uniform(0, 6)),
            })
        for fi in range(n_frames):
            frame = np.zeros((H, W, 3), dtype=np.float64)
            render_wind(frame, fi / FPS, streams, flecks, W, H, sx=1.0)
            wr.write(finalize(frame))
    elif kind == "водопад":
        # плотнее Niagara: несколько лент/брызг на атом, шире каскад
        ribbons, sprays = [], []
        for a in atoms:
            amp = float(a.get("amp") or 0.1)
            f = float(a.get("freq") or 400)
            x_norm = 0.28 + 0.44 * ((hash(str(f)) % 1000) / 1000)
            birth = float(a.get("birth") or 0)
            for _ in range(3 + int(amp * 6)):
                ribbons.append({
                    "birth": birth,
                    "x0": W * x_norm + float(rng.normal(0, 28)),
                    "amp": amp,
                    "v_y": (0.95 + amp) * (0.85 + 0.4 * rng.random()),
                    "thick": 8 + amp * 20,
                    "len": 220 + amp * 320,
                    "seed": float(rng.random()),
                    "phase": float(rng.uniform(0, 6)),
                })
            for _ in range(6 + int(amp * 14)):
                sprays.append({
                    "t0": birth + float(rng.uniform(0, 0.7)),
                    "x": float(rng.uniform(W * 0.18, W * 0.82)),
                    "y0": H * 0.5 + float(rng.uniform(0, H * 0.35)),
                    "amp": max(0.2, amp),
                    "phase": float(rng.uniform(0, 6)),
                })
        for fi in range(n_frames):
            frame = np.zeros((H, W, 3), dtype=np.float64)
            render_waterfall(frame, fi / FPS, ribbons, sprays, W, H)
            wr.write(finalize(frame, glow_sigma=2.4, mix=0.22))
    else:
        # огонь: языки из трека яркости клипа (PASS), не случайные эллипсы
        track = {"points": [], "wh": [W, H]}
        if clip is not None and clip.exists():
            track = ogon.flame_track(str(clip), t0, SEG)
        tw, th = track.get("wh") or [480, 640]
        base_y, base_x = int(H * 0.78), int(W * 0.5)
        tongues = []
        for i, a in enumerate(atoms):
            amp = float(a.get("amp") or 0.1)
            size = float(a.get("size") or 0.12)
            birth = float(a.get("birth") or 0)
            xy = ogon.nearest_xy(track, birth)
            if xy and len(xy) == 2:
                x = float(np.clip(xy[0] / max(tw, 1) * W, 60, W - 60))
                y = float(np.clip(xy[1] / max(th, 1) * H, H * 0.42, H * 0.9))
            else:
                x = float(np.clip(base_x + rng.normal(0, 70), 80, W - 80))
                y = float(base_y + rng.integers(-12, 12))
            tongues.append({
                "t": birth,
                "x": x,
                "y": y,
                "amp": amp,
                "size": size,
                "v_up": float(np.clip(0.2 + 0.7 * amp + 0.15 * size, 0.15, 1.2)),
                "phase": float(i * 0.37),
                "spark": amp > 0.08 or float(a.get("freq") or 0) > 2500,
            })
        # если атомов мало — дублируем языки по фазе треска
        if len(tongues) < 18:
            extra = []
            for i, tg in enumerate(list(tongues)):
                for k in range(2):
                    e = dict(tg)
                    e["t"] = (tg["t"] + 0.15 * (k + 1)) % SEG
                    e["x"] = float(np.clip(tg["x"] + rng.normal(0, 35), 60, W - 60))
                    e["phase"] = tg["phase"] + 1.1 * (k + 1)
                    extra.append(e)
            tongues.extend(extra)
        for fi in range(n_frames):
            t = fi / FPS
            frame = np.zeros((H, W, 3), dtype=np.float64)
            yy = np.linspace(0, 1, H)[:, None]
            frame[:, :, 0] = 2 + 6 * yy
            frame[:, :, 1] = 1 + 4 * yy
            frame[:, :, 2] = 4 + 14 * yy
            add_blob(frame, base_x, base_y + 12, 150, 40, (8, 25, 55), 0.45)
            for tg in tongues:
                age = t - tg["t"]
                # цикличность: костёр непрерывен
                if age < -0.05:
                    age += SEG
                if age > SEG:
                    age -= SEG
                if -0.02 <= age <= 1.9:
                    life = math.exp(-age * (0.5 + 0.35 * (1 - tg["amp"])))
                    if life >= 0.04:
                        rise = min(1.0, age * 2.6 + 0.1)
                        h = (130 + tg["size"] * 340 + tg["amp"] * 160) * rise * (0.55 + 0.45 * life) * (0.75 + 0.55 * tg["v_up"])
                        wob = 12 * math.sin(age * 20 + tg["phase"])
                        tip = 20 * math.sin(age * 29 + tg["phase"] * 1.7)
                        for j in range(8):
                            frac = (j + 0.5) / 8
                            width = (26 + tg["size"] * 55) * (1.15 - 0.95 * frac) * life
                            cy = tg["y"] - h * frac
                            cx = tg["x"] + wob * (1 - 0.4 * frac) + tip * frac * frac
                            if frac < 0.35:
                                bgr = (18, 60 + int(90 * frac), 230)
                            elif frac < 0.7:
                                tt = (frac - 0.35) / 0.35
                                bgr = (28, int(100 + 110 * tt), int(255 - 35 * tt))
                            else:
                                tt = (frac - 0.7) / 0.3
                                bgr = (int(190 * tt), int(225 + 30 * tt), 255)
                            add_blob(
                                frame, cx, cy, max(3, width * 0.55), max(4, width * 0.95),
                                bgr, (0.55 + 0.45 * tg["amp"]) * life * 0.95,
                            )
                        add_blob(
                            frame, tg["x"] + wob * 0.5, tg["y"] - h * 0.32,
                            9 + tg["size"] * 14, 16 + tg["size"] * 20,
                            (200, 240, 255), 0.75 * life * tg["amp"],
                        )
                if tg["spark"] and 0 <= (age if age >= 0 else age + SEG) <= 0.55:
                    aa = age if age >= 0 else age + SEG
                    fade = math.exp(-aa * 5.5)
                    sy = tg["y"] - (120 + 300 * tg["v_up"]) * aa
                    sx = tg["x"] + 28 * math.sin(aa * 48 + tg["phase"])
                    add_blob(frame, sx, sy, 2.4, 3.8, (190, 235, 255), fade * min(1, 0.55 + tg["amp"]))
            wr.write(finalize(frame, glow_sigma=3.2, mix=0.3))
    wr.release()


def remux(silent: Path, wav: Path, clip: Path, t0: float, out: Path) -> None:
    vis = out / "B_из_атомов_видео_звук.mp4"
    live = out / "A_живое_видео_звук.mp4"
    subprocess.run(
        [FF, "-y", "-i", str(silent), "-i", str(wav), "-filter:a", "volume=2.0",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(vis)],
        capture_output=True,
    )
    subprocess.run(
        [FF, "-y", "-ss", str(t0), "-t", str(SEG), "-i", str(clip),
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(live)],
        capture_output=True,
    )
    subprocess.run(
        [FF, "-y", "-ss", "0.5", "-i", str(live), "-frames:v", "1", "-update", "1",
         str(out / "still_живое.jpg")],
        capture_output=True,
    )
    subprocess.run(
        [FF, "-y", "-ss", "0.5", "-i", str(vis), "-frames:v", "1", "-update", "1",
         str(out / "still_атомы.jpg")],
        capture_output=True,
    )


def one(ex: dict) -> dict:
    out = КОРЕНЬ / ex["out"]
    out.mkdir(parents=True, exist_ok=True)
    clip = КОРЕНЬ / ex["clip"]
    full_wav = out / "full.wav"
    extract_full(clip, full_wav)
    full = load_wav(full_wav)
    i0, real, win = pick_window(full, ex["kind"])
    t0 = i0 / SR
    write_wav(out / "эталон.wav", real)
    atoms = noise.sanitize(атомизировать(real, SR))
    n_noise = sum(
        1 for a in atoms if a.get("фаза") == "шум" or float(a.get("harmonicity") or 0) < 0.12
    )
    if n_noise < 24 or ex["kind"] in ("водопад", "дождь", "река", "огонь"):
        extra = psd_filterbank_atoms(real, SR, n_max=40 if ex["kind"] == "водопад" else 36)
        atoms = noise.sanitize(atoms + extra)
    if ex["kind"] == "огонь":
        y, snap = synth_fire(atoms, real)
    elif ex["kind"] == "водопад":
        y, snap = synth_fall(atoms, real)
    else:
        y, snap = synth_rustle(atoms, real)
    write_wav(out / "сборка_чистая.wav", y)
    silent = out / "_silent.mp4"
    render_kind(ex["kind"], atoms, silent, clip=clip, t0=t0)
    remux(silent, out / "сборка_чистая.wav", clip, t0, out)
    rec = {
        "id": ex["id"],
        "имя": ex["имя"],
        "clip": ex["clip"],
        "clip_имя": ex["имя_клипа"],
        "t0": round(t0, 4),
        "n_atoms": len(atoms),
        "window": win,
        "band": snap.get("band"),
        "flatness": snap.get("flatness"),
        "centroid_hz": snap.get("centroid_hz"),
        "alignment": "single_live_clip",
        "NCC": 1.0,
        "A": "/" + ex["out"] + "/A_живое_видео_звук.mp4",
        "B": "/" + ex["out"] + "/B_из_атомов_видео_звук.mp4",
    }
    (out / "meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("id", "t0", "n_atoms", "band", "flatness")}, ensure_ascii=False))
    return rec


def write_hub(rows: list[dict]) -> None:
    hub = КОРЕНЬ / "выход/атомная_визуализация/E_все_живое_vs_атомы.html"
    extra = ["<section id='примеры2'>",
             "<h2>Ещё примеры (другие клипы)</h2>",
             "<p class='meta'>PASS-пакеты не тронуты. Каждый пример — свой video_live_02 + родной звук.</p>"]
    for r in rows:
        if str(r.get("id") or "").startswith("reka_"):
            continue
        extra.append(f"""
<div class="box">
<h3>{r['имя']} · {r['clip_имя']}</h3>
<p class="meta">t0={r['t0']}с · атомов {r['n_atoms']} · band {r.get('band')}</p>
<div class="grid">
  <div><p><b>A. Живое</b></p>
    <img src="{r['A'].replace('A_живое_видео_звук.mp4','still_живое.jpg')}"/>
    <video controls src="{r['A']}"></video></div>
  <div><p><b>B. Из атомов</b></p>
    <img src="{r['B'].replace('B_из_атомов_видео_звук.mp4','still_атомы.jpg')}"/>
    <video controls src="{r['B']}"></video></div>
</div>
</div>""")
    extra.append("</section>")
    t = hub.read_text(encoding="utf-8")
    if "id='примеры2'" in t or 'id="примеры2"' in t:
        # replace old block
        import re
        t = re.sub(r"<section id='примеры2'>.*?</section>", "", t, flags=re.S)
    if 'href="#примеры2"' not in t:
        t = t.replace(
            '<a href="#grom">6 Гром</a>',
            '<a href="#grom">6 Гром</a>\n  <a href="#примеры2">ещё примеры</a>',
        )
    t = t.replace("</body>", "\n".join(extra) + "\n</body>")
    hub.write_text(t, encoding="utf-8")


def main() -> int:
    want = set(sys.argv[1:])
    todo = [ex for ex in EXAMPLES if not want or ex["id"] in want]
    rows = [one(ex) for ex in todo]
    report = КОРЕНЬ / "отчёты/вторые_примеры_live.json"
    if report.exists() and want:
        old = json.loads(report.read_text(encoding="utf-8"))
        by_id = {r["id"]: r for r in old.get("examples") or [] if not str(r.get("id") or "").startswith("reka_")}
        for r in rows:
            by_id[r["id"]] = r
        rows = [by_id[k] for k in by_id]
    rows = [r for r in rows if not str(r.get("id") or "").startswith("reka_")]
    write_hub(rows)
    report.write_text(
        json.dumps({"дата": date.today().isoformat(), "examples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Пересобрать визуалы ветер/водопад/река с разведённой материей v2."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "scripts")]

import материя_образа as M  # noqa: E402

FF = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
if not os.path.isfile(FF):
    FF = "ffmpeg"
FPS = 30.0


def remux(silent, wav, out):
    subprocess.run(
        [FF, "-y", "-i", silent, "-i", wav, "-filter:a", "volume=2.0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-shortest", out],
        capture_output=True,
    )


def patch_matter(pkg, kind):
    mid = M.matter_id(kind)
    for a in pkg["atoms"]:
        ren = a.setdefault("образ_причины", {}).setdefault("рендер", {})
        ren["материя"] = mid["материя"]
        ren["тип"] = mid["тип"]
        ren["запрещено_материя"] = mid["запрещено_материя"]
        ne = a["образ_причины"].get("не_есть") or ""
        if "одинаковый blob" not in ne:
            a["образ_причины"]["не_есть"] = (ne + "; одинаковый blob-chain с соседями").strip("; ")


def wind_visual(pkg):
    W, H = 1280, 720
    atoms = pkg["atoms"]
    births = [float(a["birth"]) for a in atoms]
    dur = max(2.6, max(births) + 0.8 if births else 2.6)
    n_frames = int(dur * FPS)
    rng = np.random.default_rng(7)
    sx = float((pkg.get("alignment") or {}).get("flow", {}).get("sign_x") or 1.0)
    streams, flecks = [], []
    for a in atoms:
        core = a["звук_ядро"]["ядро"]
        zak = a["образ_причины"]["закон"]
        amp = float(core.get("amp") or 0.1)
        size = float(core.get("size") or 0.1)
        v_x = float(zak.get("v_x") or 0.8)
        y_norm = float(zak.get("y_norm") or 0.4)
        birth = float(a["birth"])
        for _ in range(4 + int(amp * 6)):
            streams.append({
                "birth": birth,
                "y0": H * (0.15 + 0.6 * y_norm) + float(rng.normal(0, 22)),
                "phase": float(rng.uniform(0, 6.28)),
                "amp": amp, "v_x": abs(v_x) * (0.7 + 0.5 * rng.random()),
                "thick": 1.2 + size * 6 + amp * 4,
                "wave": 14 + size * 28 + amp * 18,
                "len": 200 + size * 380 + amp * 180,
                "seed": float(rng.random()),
            })
        if amp > 0.04:
            for _ in range(6 + int(amp * 10)):
                flecks.append({
                    "t0": birth + float(rng.uniform(0, 0.5)),
                    "y": float(rng.uniform(H * 0.15, H * 0.8)),
                    "x0": float(rng.uniform(-20, W * 0.15)) if sx > 0 else float(rng.uniform(W * 0.85, W + 20)),
                    "v": (260 + amp * 420) * sx,
                    "amp": amp, "phase": float(rng.uniform(0, 6)),
                })
    vis = os.path.join(КОРЕНЬ, "выход", "атомы_полные_ветер", "визуал")
    silent = os.path.join(vis, "_silent.mp4")
    wav = os.path.join(КОРЕНЬ, "выход", "причина_ветер", "калибр_оси", "сборка_чистая.wav")
    out = os.path.join(vis, "ветер_из_атомов.mp4")
    preview = os.path.join(vis, "preview.jpg")
    os.makedirs(vis, exist_ok=True)
    wr = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fi in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.float64)
        M.render_wind(frame, fi / FPS, streams, flecks, W, H, sx=sx)
        out_f = M.finalize(frame, glow_sigma=1.6, mix=0.12)
        wr.write(out_f)
        if fi == int(0.4 * FPS):
            cv2.imwrite(preview, out_f)
    wr.release()
    remux(silent, wav, out)
    patch_matter(pkg, "ветер")
    json.dump(pkg, open(os.path.join(КОРЕНЬ, "выход", "атомы_полные_ветер", "атомы_звук_образ.json"), "w"), ensure_ascii=False)
    print("ветер OK", out)


def waterfall_visual(pkg):
    # detect size from existing or default landscape Niagara
    W, H = 1280, 720
    atoms = pkg["atoms"]
    births = [float(a["birth"]) for a in atoms]
    dur = max(2.6, max(births) + 0.9 if births else 2.6)
    n_frames = int(dur * FPS)
    rng = np.random.default_rng(11)
    ribbons, sprays = [], []
    for a in atoms:
        zak = a["образ_причины"]["закон"]
        amp = float(a["звук_ядро"]["ядро"].get("amp") or 0.1)
        v_y = float(zak.get("v_y") or 1.0)
        x_norm = float(zak.get("x_norm") or 0.5)
        spray = float(zak.get("spray") or 0.25)
        birth = float(a["birth"])
        for _ in range(2 + int(amp * 5)):
            ribbons.append({
                "birth": birth,
                "x0": W * (0.25 + 0.5 * x_norm) + float(rng.normal(0, 22)),
                "phase": float(rng.uniform(0, 6)),
                "v_y": v_y * (0.85 + 0.35 * rng.random()),
                "thick": 4.0 + amp * 16,
                "len": 160 + amp * 300,
                "amp": amp, "seed": float(rng.random()),
            })
        for _ in range(4 + int(max(spray, 0.2) * 12)):
            sprays.append({
                "t0": birth + float(rng.uniform(0, 0.6)),
                "x": float(rng.uniform(W * 0.25, W * 0.75)),
                "y0": H * 0.55 + float(rng.uniform(0, H * 0.3)),
                "amp": max(0.15, spray) * (0.5 + amp),
                "phase": float(rng.uniform(0, 6)),
            })
    vis = os.path.join(КОРЕНЬ, "выход", "атомы_полные_водопад", "визуал")
    silent = os.path.join(vis, "_silent.mp4")
    wav = os.path.join(КОРЕНЬ, "выход", "причина_водопад", "калибр_оси", "сборка_чистая.wav")
    out = os.path.join(vis, "водопад_из_атомов.mp4")
    preview = os.path.join(vis, "preview.jpg")
    os.makedirs(vis, exist_ok=True)
    wr = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fi in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.float64)
        M.render_waterfall(frame, fi / FPS, ribbons, sprays, W, H)
        out_f = M.finalize(frame, glow_sigma=2.2, mix=0.2)
        wr.write(out_f)
        if fi == int(0.4 * FPS):
            cv2.imwrite(preview, out_f)
    wr.release()
    remux(silent, wav, out)
    patch_matter(pkg, "водопад")
    json.dump(pkg, open(os.path.join(КОРЕНЬ, "выход", "атомы_полные_водопад", "атомы_звук_образ.json"), "w"), ensure_ascii=False)
    print("водопад OK", out)


def river_visual(pkg):
    W, H = 1280, 720
    atoms = pkg["atoms"]
    births = [float(a["birth"]) for a in atoms]
    dur = max(2.6, max(births) + 0.8 if births else 2.6)
    n_frames = int(dur * FPS)
    rng = np.random.default_rng(13)
    sx = float((pkg.get("alignment") or {}).get("flow", {}).get("sign_x") or 1.0)
    currents, foam = [], []
    rocks = []  # круги/диски камней — FAIL у автора («какой-то круг»)
    for a in atoms:
        zak = a["образ_причины"]["закон"]
        amp = float(a["звук_ядро"]["ядро"].get("amp") or 0.1)
        v_x = float(zak.get("v_x") or 0.8)
        y_norm = float(zak.get("y_norm") or 0.5)
        birth = float(a["birth"])
        # много тонких токов в полосе русла
        # y равномерно по всему кадру (не кластер в середине = ложный овал)
        for _ in range(5 + int(amp * 8)):
            y_base = float(rng.uniform(0.04, 0.96))
            currents.append({
                "birth": birth,
                "y0": H * y_base + float(rng.normal(0, 4)),
                "v_x": abs(v_x) * (0.7 + 0.5 * rng.random()),
                "amp": amp,
                "thick": 0.8 + amp * 2.2,
                "len": 280 + amp * 340,
                "amp_wave": 2.5 + amp * 7,
                "phase": float(rng.uniform(0, 6)),
                "seed": float(rng.random()),
            })
        if amp > 0.05:
            for _ in range(6 + int(amp * 10)):
                foam.append({
                    "t0": birth + float(rng.uniform(0, 0.5)),
                    "y": H * float(rng.uniform(0.04, 0.96)) + float(rng.normal(0, 5)),
                    "x0": float(rng.uniform(-30, W * 0.1)) if sx > 0 else float(rng.uniform(W * 0.9, W + 30)),
                    "v": (220 + amp * 380) * sx,
                    "amp": amp,
                    "phase": float(rng.uniform(0, 6)),
                })
    vis = os.path.join(КОРЕНЬ, "выход", "атомы_полные_река", "визуал")
    silent = os.path.join(vis, "_silent.mp4")
    wav = os.path.join(КОРЕНЬ, "выход", "причина_река", "калибр_оси", "сборка_чистая.wav")
    out = os.path.join(vis, "река_из_атомов.mp4")
    preview = os.path.join(vis, "preview.jpg")
    os.makedirs(vis, exist_ok=True)
    wr = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fi in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.float64)
        M.render_river(frame, fi / FPS, currents, foam, rocks, W, H, sx=sx)
        out_f = M.finalize(frame, glow_sigma=0.0, mix=0.0)  # без glow — иначе «линза»
        wr.write(out_f)
        if fi == int(0.4 * FPS):
            cv2.imwrite(preview, out_f)
    wr.release()
    remux(silent, wav, out)
    patch_matter(pkg, "река")
    # уточнить материю v3
    for a in pkg["atoms"]:
        a["образ_причины"]["рендер"]["материя"] = "current_filaments_foam"
        a["образ_причины"]["рендер"]["тип"] = "river_current_v3"
        a["образ_причины"]["не_есть"] = "эллипс-линза, тёмные круги-камни, пейзаж долины, копия FALL/CURVE, W-scale gaussian"
    json.dump(pkg, open(os.path.join(КОРЕНЬ, "выход", "атомы_полные_река", "атомы_звук_образ.json"), "w"), ensure_ascii=False)
    print("река OK", out)


def main():
    wind_visual(json.load(open(os.path.join(КОРЕНЬ, "выход", "атомы_полные_ветер", "атомы_звук_образ.json"))))
    waterfall_visual(json.load(open(os.path.join(КОРЕНЬ, "выход", "атомы_полные_водопад", "атомы_звук_образ.json"))))
    river_visual(json.load(open(os.path.join(КОРЕНЬ, "выход", "атомы_полные_река", "атомы_звук_образ.json"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

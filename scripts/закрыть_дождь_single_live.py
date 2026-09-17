# -*- coding: utf-8 -*-
"""Дождь — single_live_clip (долг axes_decoupled).

Клип: video_live_01 (Averse de pluie) — родной звук окна max(n_шум).
Старый пакет атомы_полные_дождь (7350, axes_decoupled) НЕ трогаем.
Новый пакет: атомы_полные_дождь_live.

Закон: particle_scale_impulse (капля→удар).
Материя: drop_streak_impact_crown (не река/ветер/водопад).

Запуск: python3 scripts/закрыть_дождь_single_live.py
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
import материя_образа as M  # noqa: E402

SR = 22050
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
CLIP = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
CLIP_REL = "данные/стихии_живые/dozhd/video_live_01.mp4"
CLIP_NAME = "Averse de pluie"

OUT_LIVE = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "live_sync")
OUT_ETALON = os.path.join(OUT_LIVE, "dozhd_live_01_etalon.wav")
OUT_FULL = os.path.join(OUT_LIVE, "video_live_01_full.wav")
OUT_META = os.path.join(OUT_LIVE, "sync_meta.json")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "калибр_оси_live")
OUT_WAV = os.path.join(OUT_DIR, "сборка_чистая.wav")
OUT_BASE = os.path.join(OUT_DIR, "база_шум_атомы.wav")
PKG_DIR = os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь_live")
OUT_PKG = os.path.join(PKG_DIR, "атомы_звук_образ.json")
OUT_CROSSES = os.path.join(PKG_DIR, "кресты.json")
VIS_DIR = os.path.join(PKG_DIR, "визуал")
OUT_SILENT = os.path.join(VIS_DIR, "_silent.mp4")
OUT_MP4 = os.path.join(VIS_DIR, "дождь_из_атомов.mp4")
OUT_PREVIEW = os.path.join(VIS_DIR, "preview.jpg")
E_SOUND = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "E_дождь_single_live.html")
E_EYE = os.path.join(PKG_DIR, "E_визуал_из_атомов.html")
E_CMP = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "E_сравнение_живое_vs_атомы_live.html")
CMP_DIR = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "сравнение_live")
REPORT = os.path.join(КОРЕНЬ, "отчёты", "дождь_single_live.json")
GATE = os.path.join(
    os.path.dirname(КОРЕНЬ), "Тринити cursor", "ворота", "GATE_20260810_дождь_single_live.md"
)

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
        n_noise = sum(
            1 for a in atoms if a.get("фаза") == "шум" or float(a.get("harmonicity") or 0) < 0.12
        )
        flat = noise.flatness(segn)
        score = n_noise * 10 + flat * 5 + len(atoms)
        if best is None or score > best[0]:
            best = (score, i, atoms, segn, n_noise, flat)
    _, best_i, atoms_win, seg, n_noise, flat = best
    t0 = best_i / SR
    write_wav(OUT_ETALON, seg)
    subprocess.run(
        [
            _ff(), "-y", "-ss", str(t0 + 0.4), "-i", CLIP, "-frames:v", "1", "-q:v", "2",
            "-update", "1", os.path.join(OUT_LIVE, "still_etalon.jpg"),
        ],
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
        "выбор_окна": "max(n_шум+flat) — не crest; не dozhd_real",
        "n_atoms_window": len(atoms_win),
        "n_шум": n_noise,
        "flatness": round(flat, 4),
        "note": "старый пакет атомы_полные_дождь (axes_decoupled) сохранён отдельно",
    }
    json.dump(meta, open(OUT_META, "w"), ensure_ascii=False, indent=2)
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
            a["params_104"] = {
                **p103,
                **{k: float(ox[k]) for k in ("fd", "nestedness", "mod_rate", "mod_depth", "selfsim_r2")},
            }
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
        "спектр_эталон": {
            "flatness": s_et["flatness"], "centroid_hz": s_et["centroid_hz"], "bands": s_et["bands"],
        },
        "спектр_сборка": {
            "flatness": s1["flatness"], "centroid_hz": s1["centroid_hz"], "bands": s1["bands"],
        },
        "оси_эталон": {k: s_et[k] for k in ("fd", "nestedness", "mod_rate")},
        "оси_сборка": {k: s1[k] for k in ("fd", "nestedness", "mod_rate")},
        "этапы": {"шум_атомы": s0, "сборка": s1},
        "дата": date.today().isoformat(),
    }
    return atoms, score


def step_package(atoms, sync, sound_score):
    raw = построить_кресты(atoms)
    решетка = сводка_решетки(atoms, raw)
    crosses, adj = [], defaultdict(list)
    for i, c in enumerate(raw):
        ia, ib = int(c["atom_a"]), int(c["atom_b"])
        id_a, id_b = f"dozhd_live_{ia:05d}", f"dozhd_live_{ib:05d}"
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
    mid = M.matter_id("дождь")
    full = []
    rng = np.random.default_rng(7)
    ground = H * 0.82
    for i, a in enumerate(atoms):
        birth = float(a.get("birth") or 0)
        amp = float(a.get("amp") or 0.1)
        size_a = float(a.get("size") or amp * 0.8)
        freq = float(a.get("freq") or 800)
        energy_rel = size_a / max(size_med, 1e-9)
        # размер капли ∝ size; скорость падения ∝ sqrt(energy)
        r_mm = float(np.clip(0.6 + 1.8 * energy_rel, 0.5, 3.5))
        v_ms = float(np.clip(4.0 + 5.0 * math.sqrt(max(energy_rel, 0.05)), 3.5, 12.0))
        x_norm = float(np.clip(0.08 + 0.84 * ((i * 0.6180339887) % 1.0), 0.05, 0.95))
        lean = float(rng.uniform(-28, 28))
        t_clip = t0 + birth
        img = {
            "событие": "drop_impact",
            "фаза": "both",
            "полёт": {
                "звук": os.path.relpath(OUT_ETALON, КОРЕНЬ),
                "t_flight": birth,
                "t_clip": round(t_clip, 4),
            },
            "удар": {
                "video": CLIP_REL,
                "t_clip": round(t_clip, 4),
                "роль": "точка удара — закон из того же клипа (не пейзаж)",
            },
            "закон": {
                "модель": "particle_scale_impulse",
                "формула": "y+=v*t; impact → crown; r∝size, v∝√energy",
                "r_mm": round(r_mm, 4),
                "v_ms": round(v_ms, 4),
                "energy_rel": round(energy_rel, 4),
                "size_atom": size_a,
                "size_med": size_med,
                "x_norm": round(x_norm, 4),
                "lean_px": round(lean, 2),
            },
            "геометрия": {
                "video": CLIP_REL,
                "t_sec": round(t_clip, 4),
                "t_seg": birth,
                "xy": [W * x_norm, ground],
                "approx": True,
                "note": "x из hash атома; y = поверхность удара; не трек ROI пейзажа",
            },
            "рендер": {
                **mid,
                "запрещено": ["landscape_foliage_crop", "river_current", "wind_fiber", "hud"],
            },
            "не_есть": "кроп листвы/реки, струи реки, волокна ветра, HUD-кружки",
            "метод_сборки": {
                "звук_петля": sound_score["метод"],
                "анализатор": True,
                "визуал": "DROP streak + impact crown",
                "alignment": "single_live_clip",
            },
        }
        full.append({
            "id": f"dozhd_live_{i:05d}",
            "стихия": "дождь",
            "birth": birth,
            "alignment": "single_live_clip",
            "t_sec_причина": round(t_clip, 4),
            "t0_geometry_offset": t0,
            "video_причина": CLIP_REL,
            "звук_ядро": {
                "atom_ref": f"live_atomize#{i}",
                "клетка": "live_atomize_dozhd_01",
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
                "звук_в_петле": True,
                "образ_записан_в_атом": True,
                "анализатор": True,
                "single_live_clip": True,
                "params_104_на_атоме": a.get("params_104") is not None,
            },
        })

    os.makedirs(PKG_DIR, exist_ok=True)
    pkg = {
        "стихия": "дождь",
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
            "канон": "КАК_калибровать_ветер.md + ПРАВИЛО_материя_образа.md",
            "дата": date.today().isoformat(),
        },
        "note": "пакет live; старый атомы_полные_дождь (axes_decoupled, 7350) не изменён",
        "atoms": full,
    }
    json.dump(pkg, open(OUT_PKG, "w"), ensure_ascii=False)
    json.dump({"n": len(crosses), "edges": crosses, "решетка": решетка}, open(OUT_CROSSES, "w"), ensure_ascii=False)
    return pkg


def step_visual(package):
    atoms = package["atoms"]
    births = [float(a["birth"]) for a in atoms]
    dur = max(SEG_DUR, max(births) + 1.0 if births else SEG_DUR)
    n_frames = int(dur * FPS)
    drops = []
    rng = np.random.default_rng(11)
    for a in atoms:
        zak = a["образ_причины"]["закон"]
        amp = float(a["звук_ядро"]["ядро"].get("amp") or 0.1)
        # плотность: несколько капель + период повтора
        for k in range(2 + int(amp * 4)):
            drops.append({
                "birth": float(a["birth"]) + 0.03 * k + float(rng.uniform(0, 0.2)),
                "amp": max(0.15, amp * (0.65 + 0.35 * rng.random())),
                "x": float(zak["x_norm"]) * W + (k - 1.5) * 22 + float(rng.normal(0, 12)),
                "y_imp": H * 0.82 + float(rng.uniform(-8, 14)),
                "lean": float(zak.get("lean_px") or 0) + float(rng.uniform(-20, 20)),
                "period": float(rng.uniform(0.4, 0.75)),
            })
    os.makedirs(VIS_DIR, exist_ok=True)
    wr = cv2.VideoWriter(OUT_SILENT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fi in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.float64)
        M.render_rain(frame, fi / FPS, drops, W, H)
        out_f = M.finalize(frame, glow_sigma=0.6, mix=0.06)
        wr.write(out_f)
        if fi == int(0.25 * FPS):
            cv2.imwrite(OUT_PREVIEW, out_f)
    wr.release()
    subprocess.run(
        [
            _ff(), "-y", "-i", OUT_SILENT, "-i", OUT_WAV, "-filter:a", "volume=2.0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", OUT_MP4,
        ],
        capture_output=True, timeout=120,
    )


def step_compare(sync):
    os.makedirs(CMP_DIR, exist_ok=True)
    dur, t0 = float(sync["seg_dur_sec"]), float(sync["t0_sec"])
    live_win = os.path.join(CMP_DIR, "A_живое_окно.mp4")
    atom_win = os.path.join(CMP_DIR, "B_из_атомов.mp4")
    for cmd in (
        [
            _ff(), "-y", "-ss", str(t0), "-i", CLIP, "-t", str(dur),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", live_win,
        ],
        [
            _ff(), "-y", "-i", OUT_MP4, "-t", str(dur),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", atom_win,
        ],
    ):
        subprocess.run(cmd, capture_output=True, timeout=90)
    # hstack через OpenCV — tools/ffmpeg filter_complex здесь зависает
    side = os.path.join(CMP_DIR, "рядом_честный.mp4")
    ca, cb = cv2.VideoCapture(live_win), cv2.VideoCapture(atom_win)
    fps = ca.get(cv2.CAP_PROP_FPS) or FPS
    n = int(dur * fps)
    wr = cv2.VideoWriter(side, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))
    for i in range(n):
        ok1, f1 = ca.read()
        ok2, f2 = cb.read()
        if not ok1 or not ok2:
            break
        f1 = cv2.resize(f1, (640, 360))
        f2 = cv2.resize(f2, (640, 360))
        wr.write(cv2.hconcat([f1, f2]))
        if i == 12:
            cv2.imwrite(os.path.join(CMP_DIR, "still_живое.jpg"), f1)
            cv2.imwrite(os.path.join(CMP_DIR, "still_атомы.jpg"), f2)
    wr.release()
    ca.release()
    cb.release()


def step_pages(sync, score, pkg):
    os.makedirs(os.path.dirname(E_SOUND), exist_ok=True)
    se, ss = score["спектр_эталон"], score["спектр_сборка"]
    open(E_SOUND, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E ухо — дождь single_live</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.3rem 0 .9rem}}.meta{{opacity:.85}}</style></head><body>
<h1>E ухо — дождь single_live</h1>
<p class="meta">{score['дата']} · {CLIP_NAME} · t0={sync['t0_sec']}с (окно по n_шум)<br/>
atoms {score['n_atoms']} шум={score['n_фаза_шум']} · 104={score['params_104_filled']}<br/>
flatness {se['flatness']}→{ss['flatness']} · band {score['band']}<br/>
эталон = дорожка клипа (не dozhd_real)</p>
<p>Эталон</p><audio controls src="live_sync/dozhd_live_01_etalon.wav"></audio>
<p>Сборка</p><audio controls src="калибр_оси_live/сборка_чистая.wav"></audio>
<p><a href="/выход/атомы_полные_дождь_live/E_визуал_из_атомов.html" style="color:#9dceb0">→ E глаз</a> ·
<a href="/выход/причина_дождь/E_сравнение_живое_vs_атомы_live.html" style="color:#9dceb0">→ рядом</a></p>
<p><b>Cmd+Shift+R</b>. Узнаётся ли дождь (капли/пелена), не ветер/река?</p>
</body></html>"""
    )
    open(E_EYE, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E глаз — дождь DROP</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:960px;margin:2rem auto;padding:0 1rem}}
video{{width:100%;background:#000}}.meta{{opacity:.85}}</style></head><body>
<h1>E глаз — дождь DROP</h1>
<p class="meta">закон <b>particle_scale_impulse</b> · материя <b>drop_streak_impact_crown</b> · atoms {pkg['n_atoms']}<br/>
штрих полёта + корона удара — не пейзаж, не струи реки</p>
<video controls src="визуал/дождь_из_атомов.mp4"></video>
</body></html>"""
    )
    open(E_CMP, "w", encoding="utf-8").write(
        """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E сравнение — дождь live</title>
<style>
body{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:1100px;margin:1.5rem auto;padding:0 1rem}
video{width:100%;background:#000;margin:.4rem 0 1rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem} img{width:100%;background:#000}
</style></head><body>
<h1>Дождь live: живое vs из атомов</h1>
<p>Слева — окно клипа (пейзаж с дождём). Справа — материя капли→удар (не копия кадра).</p>
<video controls src="сравнение_live/рядом_честный.mp4"></video>
<div class="grid">
<div><p><b>A. Живое окно</b></p><video controls src="сравнение_live/A_живое_окно.mp4"></video></div>
<div><p><b>B. Из атомов</b></p><video controls src="сравнение_live/B_из_атомов.mp4"></video></div>
</div>
<div class="grid">
<div><img src="сравнение_live/still_живое.jpg"/></div>
<div><img src="сравнение_live/still_атомы.jpg"/></div>
</div>
</body></html>"""
    )


def main() -> int:
    print("=== A0 ===", flush=True)
    sync, real = step_a0()
    print("t0", sync["t0_sec"], "n_шум", sync["n_шум"], "flat", sync["flatness"], flush=True)
    print("=== sound ===", flush=True)
    atoms, score = step_sound(real)
    print("band", score["band"], "n", score["n_atoms"], flush=True)
    print("=== package/visual/E ===", flush=True)
    pkg = step_package(atoms, sync, score)
    step_visual(pkg)
    step_compare(sync)
    step_pages(sync, score, pkg)
    json.dump(
        {"sync": sync, "sound_score": score, "n_atoms": pkg["n_atoms"]},
        open(REPORT, "w"),
        ensure_ascii=False,
        indent=2,
    )
    open(GATE, "w", encoding="utf-8").write(
        f"""# GATE — дождь single_live

> {score['дата']} · OPEN → E автора

- клип: `{CLIP_REL}` ({CLIP_NAME})
- t0={sync['t0_sec']}с · окно max(n_шум)
- эталон = дорожка клипа (не `dozhd_real`) · NCC=1.0
- звук: шум из атомов · band={score['band']}
- закон: `particle_scale_impulse` · материя: `drop_streak_impact_crown`
- пакет: `выход/атомы_полные_дождь_live/` (старый 7350 axes_decoupled сохранён)
- E ухо: `выход/причина_дождь/E_дождь_single_live.html`
- E глаз: `выход/атомы_полные_дождь_live/E_визуал_из_атомов.html`
"""
    )
    print("OK", OUT_MP4, "band", score["band"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

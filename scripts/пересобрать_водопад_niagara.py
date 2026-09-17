# -*- coding: utf-8 -*-
"""Водопад — референс Niagara Horseshoe (Commons, родной звук).

Источник: File:NiagaraKanadaHufeisenfall2006Mai.ogv (CC, Hedwig Storch, 2006)
→ данные/стихии_живые/vodopad/video_live_03_niagara.mp4

Запуск: python3 scripts/пересобрать_водопад_niagara.py
"""
from __future__ import annotations

import json
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

import закрыть_водопад_single_live as v  # noqa: E402
import калибр_ветер_шум_из_атомов as noise  # noqa: E402
from ядро.атомизация import атомизировать  # noqa: E402
from atoms_full103 import analyze_full_103  # noqa: E402
from оси import оси_звука  # noqa: E402
import пересобрать_водопад_live01 as live01  # noqa: E402

SR = 22050
CLIP = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "vodopad", "video_live_03_niagara.mp4")
CLIP_REL = "данные/стихии_живые/vodopad/video_live_03_niagara.mp4"
CLIP_NAME = "Niagara Horseshoe Falls (boat, 2006)"
SOURCE = {
    "commons": "File:NiagaraKanadaHufeisenfall2006Mai.ogv",
    "author": "Hedwig Storch",
    "url": "https://commons.wikimedia.org/wiki/File:NiagaraKanadaHufeisenfall2006Mai.ogv",
    "license": "CC (see Commons page)",
}


def main() -> int:
    v.CLIP, v.CLIP_REL, v.CLIP_NAME = CLIP, CLIP_REL, CLIP_NAME
    v.OUT_ETALON = os.path.join(v.OUT_LIVE, "vodopad_niagara_etalon.wav")
    v.OUT_FULL = os.path.join(v.OUT_LIVE, "video_live_03_niagara_full.wav")
    # Niagara — горизонтальный кадр
    v.W, v.H = 1280, 720

    os.makedirs(v.OUT_LIVE, exist_ok=True)
    open(os.path.join(v.OUT_LIVE, "SOURCE_niagara.json"), "w").write(
        json.dumps(SOURCE, ensure_ascii=False, indent=2)
    )

    print("=== A0 Niagara ===", flush=True)
    subprocess.run(
        [v._ff(), "-y", "-i", CLIP, "-vn", "-ac", "1", "-ar", str(SR), v.OUT_FULL],
        capture_output=True,
        check=True,
    )
    full, _ = v.load_wav(v.OUT_FULL)
    win = int(v.SEG_DUR * SR)
    # окно с max flatness×rms (гул каскада)
    best_i, best_s = 0, -1.0
    hop = int(0.5 * SR)
    for i in range(0, max(1, len(full) - win), hop):
        seg = full[i : i + win]
        segn = seg / (np.max(np.abs(seg)) + 1e-12) * 0.9
        flat = noise.flatness(segn)
        rms = float(np.sqrt(np.mean(segn ** 2)))
        s = flat * 2.0 + rms
        if s > best_s:
            best_s, best_i = s, i
    t0 = best_i / SR
    real = full[best_i : best_i + win]
    real = real / (np.max(np.abs(real)) + 1e-12) * 0.9
    v.write_wav(v.OUT_ETALON, real)
    still = os.path.join(v.OUT_LIVE, "still_etalon.jpg")
    subprocess.run(
        [v._ff(), "-y", "-ss", str(t0 + 0.5), "-i", CLIP, "-frames:v", "1", "-q:v", "2",
         "-update", "1", still],
        capture_output=True,
    )

    track = noise.sanitize(атомизировать(real, SR))
    n_noise = sum(1 for a in track if a.get("фаза") == "шум" or float(a.get("harmonicity") or 0) < 0.12)
    print("track", len(track), "шум", n_noise, "flat", round(noise.flatness(real), 4), flush=True)
    if n_noise < 8:
        atoms = live01.psd_filterbank_atoms(real, SR, n_max=40)
        atom_src = "psd_filterbank (+ track sparse)"
        # merge unique freqs from track noise if any
        atoms = atoms + [a for a in track if a.get("фаза") == "шум"]
    else:
        atoms = track
        atom_src = "атомизация"

    sync = {
        "дата": date.today().isoformat(),
        "clip": CLIP_REL,
        "clip_имя": CLIP_NAME,
        "t0_sec": round(t0, 4),
        "seg_dur_sec": v.SEG_DUR,
        "NCC_segment_vs_clip_window": 1.0,
        "alignment": "single_live_clip",
        "etalon_wav": os.path.relpath(v.OUT_ETALON, КОРЕНЬ),
        "выбор_окна": "max(flat×2+rms)",
        "n_atoms_window": len(atoms),
        "n_шум": sum(1 for a in atoms if a.get("фаза") == "шум"),
        "flatness": round(noise.flatness(real), 4),
        "atom_source": atom_src,
        "source": SOURCE,
        "note": "автор: взять Ниагару; live_02 струйка REJECT; live_01 Syratu сменён",
        "rejected": ["video_live_02.mp4", "video_live_01.mp4 (заменён по просьбе Niagara)"],
    }
    json.dump(sync, open(v.OUT_META, "w"), ensure_ascii=False, indent=2)

    print("=== sound ===", flush=True)
    os.makedirs(v.OUT_DIR, exist_ok=True)
    n = len(real)
    y0 = noise.синтез_шум_из_атомов(atoms, n, SR)
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12) * 0.9
    v.write_wav(v.OUT_BASE, y0)
    y1 = noise.apply_env(y0[:n], real, SR, blend=0.45)
    y1 = noise.eq_full_psd(y1, real, SR, gmax=2.4)
    s0, s_et = noise.snap("шум_атомы", y0, real), noise.snap("эталон", real, real)
    s1 = noise.snap("сборка", y1, real)
    if s0["band"] > s1["band"] + 0.03:
        y1 = noise.apply_env(y0[:n], real, SR, blend=0.4)
        y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
        s1 = noise.snap("сборка", y1, real)
    else:
        y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
    v.write_wav(v.OUT_WAV, y1)

    n104 = 0
    half = int(0.08 * SR)
    for a in atoms:
        mid = int(0.45 * float(a.get("lifetime") or v.SEG_DUR) * SR)
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

    score = {
        "метод": f"{atom_src} → шум-filterbank → EQ",
        "анализатор": True,
        "рычаг": "шум_из_атомов",
        "референс": CLIP_NAME,
        "source": SOURCE,
        "n_atoms": len(atoms),
        "n_фаза_шум": sum(1 for a in atoms if a.get("фаза") == "шум"),
        "params_104_filled": n104,
        "band": s1["band"],
        "stft": s1["stft"],
        "спектр_эталон": {"flatness": s_et["flatness"], "centroid_hz": s_et["centroid_hz"], "bands": s_et["bands"]},
        "спектр_сборка": {"flatness": s1["flatness"], "centroid_hz": s1["centroid_hz"], "bands": s1["bands"]},
        "оси_эталон": {k: s_et[k] for k in ("fd", "nestedness", "mod_rate")},
        "оси_сборка": {k: s1[k] for k in ("fd", "nestedness", "mod_rate")},
        "этапы": {"шум_атомы": s0, "сборка": s1},
        "дата": date.today().isoformat(),
        "E_вход": "взять за основу Ниагарский водопад",
    }

    print("=== package / visual ===", flush=True)
    pkg = v.step_package(atoms, sync, score)
    pkg["alignment"].update({
        "clip": CLIP_REL, "clip_имя": CLIP_NAME, "video_ось": CLIP_REL,
        "звук_ось": os.path.relpath(v.OUT_ETALON, КОРЕНЬ),
    })
    for a in pkg["atoms"]:
        a["video_причина"] = CLIP_REL
        a["звук_ядро"]["звук_путь"] = os.path.relpath(v.OUT_ETALON, КОРЕНЬ)
        a["звук_ядро"]["клетка"] = "live_atomize_vodopad_niagara"
        img = a["образ_причины"]
        img["падение"]["звук"] = os.path.relpath(v.OUT_ETALON, КОРЕНЬ)
        img["каскад"]["video"] = CLIP_REL
        img["геометрия"]["video"] = CLIP_REL
        img["метод_сборки"]["звук_петля"] = score["метод"]
        img["не_есть"] = "кроп пейзажа Ниагары, HUD, чужой клип, струйка live_02"
    json.dump(pkg, open(v.OUT_PKG, "w"), ensure_ascii=False)

    v.step_visual(pkg)
    v.step_compare(sync)
    v.step_pages(sync, score, pkg)

    # fix E paths/labels
    ear = open(v.E_SOUND, encoding="utf-8").read()
    ear = ear.replace("vodopad_live_02_etalon.wav", "vodopad_niagara_etalon.wav")
    ear = ear.replace("vodopad_live_01_etalon.wav", "vodopad_niagara_etalon.wav")
    ear = ear.replace("клип live_02", "Niagara Horseshoe")
    ear = ear.replace("клип live_01 · Cascade de Syratu", "Niagara Horseshoe Falls")
    if "Niagara" not in ear.split("h1")[0] + ear[:500]:
        pass
    open(v.E_SOUND, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E ухо — водопад Niagara</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.3rem 0 .9rem}}.meta{{opacity:.85}}</style></head><body>
<h1>E ухо — водопад (Niagara)</h1>
<p class="meta">{score['дата']} · <b>{CLIP_NAME}</b><br/>
Commons: NiagaraKanadaHufeisenfall2006Mai · родной звук<br/>
atoms {score['n_atoms']} шум={score['n_фаза_шум']} · 104={score['params_104_filled']}<br/>
flatness {score['спектр_эталон']['flatness']}→{score['спектр_сборка']['flatness']}
· band {score['band']}<br/>
REJECT: live_02 струйка · live_01 Syratu (заменён по просьбе)</p>
<p>Эталон</p><audio controls src="live_sync/vodopad_niagara_etalon.wav"></audio>
<p>Сборка из атомов</p><audio controls src="калибр_оси/сборка_чистая.wav"></audio>
<p><a href="/выход/атомы_полные_водопад/E_визуал_из_атомов.html" style="color:#9dceb0">→ E глаз</a> ·
<a href="/выход/причина_водопад/E_сравнение_живое_vs_атомы.html" style="color:#9dceb0">→ рядом</a></p>
<p><b>Cmd+Shift+R</b>. Узнаётся ли гул Ниагары?</p>
</body></html>"""
    )
    open(v.E_EYE, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E глаз — Niagara FALL</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:960px;margin:2rem auto;padding:0 1rem}}
video{{width:100%;background:#000}}.meta{{opacity:.85}}</style></head><body>
<h1>E глаз — Niagara FALL</h1>
<p class="meta">закон <b>gravity_fall_spray</b> · atoms {pkg['n_atoms']}<br/>
не кроп живой Ниагары — падение + дымка из атомов</p>
<video controls src="визуал/водопад_из_атомов.mp4"></video>
<p><b>Cmd+Shift+R</b></p>
</body></html>"""
    )
    open(v.E_CMP, "w", encoding="utf-8").write(
        """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E сравнение — Niagara</title>
<style>
body{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:1100px;margin:1.5rem auto;padding:0 1rem}
video{width:100%;background:#000;margin:.4rem 0 1rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
img{width:100%;background:#000}
</style></head><body>
<h1>Niagara: живое vs из атомов</h1>
<p>Horseshoe Falls · single_live · шум из атомов + FALL</p>
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

    open(v.GATE, "w", encoding="utf-8").write(
        f"""# GATE — водопад Niagara

> {score['дата']} · OPEN → E · референс **Niagara Horseshoe**

| | |
|--|--|
| клип | `video_live_03_niagara.mp4` |
| источник | {SOURCE['commons']} |
| t0 | {sync['t0_sec']}с |
| atoms | {score['n_atoms']} · {atom_src} |
| band | {score['band']} |

REJECT: live_02 струйка · live_01 Syratu (по просьбе автора — Ниагара).

E: `выход/причина_водопад/E_водопад_шум_атомы.html`
"""
    )
    # meta_live update
    meta_p = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "vodopad", "meta_live.json")
    try:
        meta = json.load(open(meta_p))
    except Exception:
        meta = {"стихия": "vodopad", "источники": []}
    meta["источники"] = list(meta.get("источники") or [])
    meta["источники"].append({
        "файл": "video_live_03_niagara.mp4",
        "источник": SOURCE["commons"],
        "звук": "родной",
        "дата": date.today().isoformat(),
        "роль": "текущий single_live эталон",
    })
    meta["текущий_эталон"] = "video_live_03_niagara.mp4"
    json.dump(meta, open(meta_p, "w"), ensure_ascii=False, indent=2)

    json.dump({"sync": sync, "sound_score": score}, open(v.REPORT, "w"), ensure_ascii=False, indent=2)
    print(json.dumps({"band": score["band"], "n": score["n_atoms"], "flat_et": score["спектр_эталон"]["flatness"],
                      "flat_as": score["спектр_сборка"]["flatness"], "t0": sync["t0_sec"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

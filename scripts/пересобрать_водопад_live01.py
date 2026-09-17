# -*- coding: utf-8 -*-
"""Водопад — смена референса на video_live_01 (Cascade de Syratu).

E автора 2026-08-10: live_02 («Smoky Mountains») — плохой референс (струйка в лесу).
live_01 — узнаваемый каскад. Атомизатор на нём даёт 1×переход (узкий спектр) →
добираем filterbank-атомы из PSD-пиков **того же** эталона (не чужой клип).

Запуск: python3 scripts/пересобрать_водопад_live01.py
"""
from __future__ import annotations

import json
import os
import shutil
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

SR = 22050
CLIP = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "vodopad", "video_live_01.mp4")
CLIP_REL = "данные/стихии_живые/vodopad/video_live_01.mp4"
CLIP_NAME = "Cascade de Syratu"
REJECT = os.path.join(v.OUT_LIVE, "_rejected_live02_струйка")


def psd_filterbank_atoms(x: np.ndarray, sr: int = SR, n_max: int = 36) -> list[dict]:
    """Атомы-полосы из пиков PSD (фаза=шум), когда трек-атомизация схлопывается."""
    f, P = signal.welch(x, sr, nperseg=2048)
    peaks, _ = signal.find_peaks(P, prominence=np.percentile(P, 35) * 0.15, distance=2)
    pf, pp = f[peaks], P[peaks]
    mask = (pf > 100) & (pf < 9000)
    pf, pp = pf[mask], pp[mask]
    if len(pf) == 0:
        return []
    order = np.argsort(-pp)[:n_max]
    amp_max = float(np.max(pp[order])) + 1e-18
    dur = len(x) / sr
    atoms = []
    for rank, i in enumerate(order):
        freq = float(pf[i])
        amp = float(np.sqrt(pp[i] / amp_max))
        atoms.append({
            "birth": 0.0,
            "lifetime": round(dur, 4),
            "freq": round(freq, 2),
            "amp": round(float(np.clip(amp, 0.05, 1.0)), 4),
            "harmonicity": 0.0,
            "harmonic_index": 0,
            "phase": 0.0,
            "фаза": "шум",
            "size": round(float(np.clip(amp, 0.05, 1.0)) * 0.8, 4),
            "amp_t": [round(float(np.clip(amp, 0.05, 1.0)), 4)] * 8,
            "freq_t": [round(freq, 2)] * 8,
            "attack_ratio": 0.05,
            "decay_shape": -0.5,
            "freq_slope": 0.0,
            "_kind": "шум",
            "_источник": "psd_filterbank",
            "_rank": rank,
        })
    return atoms


def main() -> int:
    # patch module paths to live_01
    v.CLIP = CLIP
    v.CLIP_REL = CLIP_REL
    v.CLIP_NAME = CLIP_NAME
    v.OUT_ETALON = os.path.join(v.OUT_LIVE, "vodopad_live_01_etalon.wav")
    v.OUT_FULL = os.path.join(v.OUT_LIVE, "video_live_01_full.wav")

    os.makedirs(v.OUT_LIVE, exist_ok=True)
    # archive old live02 still/etalon names if present
    for name in ("vodopad_live_02_etalon.wav", "still_etalon.jpg", "sync_meta.json"):
        src = os.path.join(v.OUT_LIVE, name)
        if os.path.isfile(src) and "live_02" in open(v.OUT_META).read() if os.path.isfile(v.OUT_META) else False:
            pass
    # mark rejection
    open(os.path.join(v.OUT_LIVE, "REJECTED_live02.txt"), "w").write(
        "2026-08-10 автор: плохой референс (струйка в лесу, Smoky Mountains).\n"
        "Заменён на video_live_01 Cascade de Syratu.\n"
    )

    print("=== A0 live_01 ===", flush=True)
    subprocess.run(
        [v._ff(), "-y", "-i", CLIP, "-vn", "-ac", "1", "-ar", str(SR), v.OUT_FULL],
        capture_output=True,
        check=True,
    )
    full, _ = v.load_wav(v.OUT_FULL)
    # окно: энергия × flatness (каскад непрерывный) — середина клипа устойчивее
    win = int(v.SEG_DUR * SR)
    best_i, best_s = int(3.0 * SR), -1.0
    hop = int(0.5 * SR)
    for i in range(0, max(1, len(full) - win), hop):
        seg = full[i : i + win]
        segn = seg / (np.max(np.abs(seg)) + 1e-12) * 0.9
        flat = noise.flatness(segn)
        rms = float(np.sqrt(np.mean(segn ** 2)))
        # предпочитаем не тишину; flat у этого клипа низкий — не гнать
        s = rms * (0.5 + flat)
        if s > best_s:
            best_s, best_i = s, i
    t0 = best_i / SR
    real = full[best_i : best_i + win].copy()
    real = real / (np.max(np.abs(real)) + 1e-12) * 0.9
    v.write_wav(v.OUT_ETALON, real)
    still = os.path.join(v.OUT_LIVE, "still_etalon.jpg")
    subprocess.run(
        [v._ff(), "-y", "-ss", str(t0 + 0.5), "-i", CLIP, "-frames:v", "1", "-q:v", "2", still],
        capture_output=True,
    )

    track_atoms = noise.sanitize(атомизировать(real, SR))
    n_noise = sum(1 for a in track_atoms if a.get("фаза") == "шум" or float(a.get("harmonicity") or 0) < 0.12)
    print("track atoms", len(track_atoms), "шум", n_noise, flush=True)

    if n_noise < 5:
        atoms = psd_filterbank_atoms(real, SR, n_max=36)
        atom_src = "psd_filterbank (атомизатор=1 трек на live_01)"
        print("→ PSD filterbank atoms", len(atoms), flush=True)
    else:
        atoms = track_atoms
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
        "выбор_окна": "rms×(0.5+flat) mid-clip",
        "n_atoms_window": len(atoms),
        "n_шум": sum(1 for a in atoms if a.get("фаза") == "шум"),
        "flatness": round(noise.flatness(real), 4),
        "atom_source": atom_src,
        "note": "live_02 отвергнут автором (струйка); live_01 Cascade de Syratu",
        "rejected": "video_live_02.mp4 Smoky Mountains trickle",
    }
    with open(v.OUT_META, "w", encoding="utf-8") as f:
        json.dump(sync, f, ensure_ascii=False, indent=2)

    print("=== sound ===", flush=True)
    # reuse sound path but with our atoms
    os.makedirs(v.OUT_DIR, exist_ok=True)
    n = len(real)
    y0 = noise.синтез_шум_из_атомов(atoms, n, SR)
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12) * 0.9
    v.write_wav(v.OUT_BASE, y0)
    y1 = noise.apply_env(y0[:n], real, SR, blend=0.4)
    y1 = noise.eq_full_psd(y1, real, SR, gmax=2.2)
    sh, et = noise.band_shares(y1), noise.band_shares(real)
    if (sh["1.2-3.8k"] + sh["3.8-6k"]) < (et["1.2-3.8k"] + et["3.8-6k"]) * 0.55:
        hp = signal.sosfiltfilt(signal.butter(2, 200, btype="high", fs=SR, output="sos"), y1)
        y1 = 0.5 * hp + 0.5 * y0[:n]
        y1 = noise.eq_full_psd(y1, real, SR, gmax=1.8)
    y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
    s0 = noise.snap("шум_атомы", y0, real)
    s1 = noise.snap("сборка", y1, real)
    s_et = noise.snap("эталон", real, real)
    if s0["band"] > s1["band"] + 0.03:
        y1 = noise.apply_env(y0[:n], real, SR, blend=0.35)
        y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9
        s1 = noise.snap("сборка", y1, real)
    v.write_wav(v.OUT_WAV, y1)

    # 104
    from atoms_full103 import analyze_full_103
    from оси import оси_звука
    n104 = 0
    half = int(0.08 * SR)
    for a in atoms:
        mid = int(0.5 * float(a.get("lifetime") or 0.2) * SR)
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

    n_noise = sum(1 for a in atoms if a.get("фаза") == "шум")
    score = {
        "метод": f"{atom_src} → шум-filterbank → EQ (канон ветра)",
        "анализатор": True,
        "рычаг": "шум_из_атомов",
        "референс": CLIP_NAME,
        "rejected_ref": "live_02 струйка",
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
        "E_вход": "плохой референс live_02 → live_01",
    }

    print("=== package / visual / E ===", flush=True)
    # ensure package uses updated CLIP_REL
    pkg = v.step_package(atoms, sync, score)
    # fix package clip fields if module still had old constants baked in builder via CLIP_REL
    pkg["alignment"]["clip"] = CLIP_REL
    pkg["alignment"]["clip_имя"] = CLIP_NAME
    pkg["alignment"]["video_ось"] = CLIP_REL
    pkg["alignment"]["звук_ось"] = os.path.relpath(v.OUT_ETALON, КОРЕНЬ)
    for a in pkg["atoms"]:
        a["video_причина"] = CLIP_REL
        a["звук_ядро"]["звук_путь"] = os.path.relpath(v.OUT_ETALON, КОРЕНЬ)
        a["звук_ядро"]["клетка"] = "live_atomize_vodopad_01"
        img = a["образ_причины"]
        img["падение"]["звук"] = os.path.relpath(v.OUT_ETALON, КОРЕНЬ)
        img["каскад"]["video"] = CLIP_REL
        img["геометрия"]["video"] = CLIP_REL
        img["метод_сборки"]["звук_петля"] = score["метод"]
    with open(v.OUT_PKG, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False)

    v.step_visual(pkg)
    v.step_compare(sync)
    v.step_pages(sync, score, pkg)

    # fix E audio paths to live_01 etalon
    ear = open(v.E_SOUND, encoding="utf-8").read()
    ear = ear.replace("vodopad_live_02_etalon.wav", "vodopad_live_01_etalon.wav")
    ear = ear.replace("клип live_02", f"клип live_01 · {CLIP_NAME}")
    if "live_02 отвергнут" not in ear:
        ear = ear.replace(
            "</p>\n<p>Эталон</p>",
            "<br/><b>референс:</b> Cascade de Syratu (live_02 струйка — REJECT)</p>\n<p>Эталон</p>",
        )
    open(v.E_SOUND, "w", encoding="utf-8").write(ear)

    open(v.GATE, "w", encoding="utf-8").write(
        f"""# GATE — водопад single_live

> {score['дата']} · OPEN → E автора · **референс сменён**

## Референс

| | |
|--|--|
| **сейчас** | `video_live_01.mp4` — Cascade de Syratu |
| **REJECT** | `video_live_02.mp4` — Smoky Mountains, струйка (автор 2026-08-10) |

- t0={sync['t0_sec']}с · atoms={score['n_atoms']} ({atom_src})
- звук: шум-filterbank · band={score['band']} · flatness {score['спектр_эталон']['flatness']}→{score['спектр_сборка']['flatness']}
- образ: `gravity_fall_spray`
- E ухо: `выход/причина_водопад/E_водопад_шум_атомы.html`
- E глаз: `выход/атомы_полные_водопад/E_визуал_из_атомов.html`
"""
    )
    json.dump({"sync": sync, "sound_score": score}, open(v.REPORT, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(score, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

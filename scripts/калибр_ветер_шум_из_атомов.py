# -*- coding: utf-8 -*-
"""Ветер — честный шелест из атомов (фаза=шум → noise filterbank).

E FAIL 2026-08-06: «сборка анти-цикл ужас».
Диагноз (числа + ухо):
  - эталон flatness≈0.41 (широкополосный шум), centroid≈3.3 кГц
  - анти-цикл / синтез_из_атомов: flatness≈0.00–0.002 (синусы!), centroid≈1.6 кГц
  - атомы ветра уже фаза=шум, lifetime≈весь клип — filterbank; но
    `синтез_из_атомов` всегда рисует sin-зёрна → метал/писк, не шелест.
  - развязка ради nest убила HF (6k+ 23%→1%) — ухо важнее nest.

Рычаг: атомизация → **шум-зёрна по freq атомов** → огибающая → EQ к PSD эталона.
НЕ chase nestedness. НЕ leaf из сырого эталона как «тело».

Запуск: python3 scripts/калибр_ветер_шум_из_атомов.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import wave
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
from ядро.атомизация import атомизировать  # noqa: E402
import закрыть_ветер_single_live as w  # noqa: E402

SR = 22050
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "калибр_ветер_шум_из_атомов.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "калибр_ветер_шум_из_атомов.md")
OUT_FAIL = os.path.join(w.OUT_DIR, "сборка_anti_cycle_FAIL.wav")
OUT_PROBE_BASE = os.path.join(w.OUT_DIR, "база_шум_атомы.wav")


def flatness(x: np.ndarray, sr: int = SR) -> float:
    _, P = signal.welch(x, sr, nperseg=2048)
    P = P + 1e-18
    return float(np.exp(np.mean(np.log(P))) / np.mean(P))


def centroid_hz(x: np.ndarray, sr: int = SR) -> float:
    f, P = signal.welch(x, sr, nperseg=2048)
    return float(np.sum(f * P) / (np.sum(P) + 1e-18))


def band_shares(x: np.ndarray, sr: int = SR) -> dict:
    f, P = signal.welch(x, sr, nperseg=2048)
    tot = float(np.sum(P)) + 1e-18

    def share(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.sum(P[m]) / tot)

    return {
        "0-400": round(share(0, 400), 3),
        "400-1200": round(share(400, 1200), 3),
        "1.2-3.8k": round(share(1200, 3800), 3),
        "3.8-6k": round(share(3800, 6000), 3),
        "6k+": round(share(6000, 11000), 3),
    }


def sanitize(atoms: list[dict]) -> list[dict]:
    out = []
    for a in atoms:
        b = dict(a)
        ph = b.get("phase")
        if isinstance(ph, str):
            b["фаза"] = ph
            b["phase"] = 0.0
        elif ph is None:
            b["phase"] = 0.0
        if b.get("size") is None:
            b["size"] = float(b.get("amp") or 0.1) * 0.8
        out.append(b)
    return out


def _bw_table(freqs: list[float]) -> dict[float, float]:
    freqs = sorted(freqs)
    out = {}
    for i, f in enumerate(freqs):
        lo = freqs[i - 1] if i > 0 else f / 1.45
        hi = freqs[i + 1] if i + 1 < len(freqs) else f * 1.45
        # шире на ВЧ — шелест листьев
        scale = 1.15 if f < 1500 else (1.45 if f < 4000 else 1.8)
        bw = max(90.0, min(3200.0, (hi - lo) * scale, f * 0.7 + 120))
        out[f] = bw
    return out


def синтез_шум_из_атомов(atoms: list[dict], n: int, sr: int = SR, seed: int = 7) -> np.ndarray:
    """Каждый атом с фазой шум/низкой harm → bandpass-шум на freq, длина=lifetime."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float64)
    freqs = [float(a["freq"]) for a in atoms if float(a.get("freq") or 0) > 40]
    bws = _bw_table(freqs)
    for a in atoms:
        f0 = float(a.get("freq") or 0)
        if f0 < 40 or f0 >= sr / 2 - 80:
            continue
        harm = float(a.get("harmonicity") or 0)
        fase = a.get("фаза") or a.get("phase")
        is_noise = (isinstance(fase, str) and fase == "шум") or harm < 0.12
        birth = float(a.get("birth") or 0)
        life = float(a.get("lifetime") or 0.15)
        i0 = int(birth * sr)
        L = int(max(0.08, life) * sr)
        i1 = min(n, i0 + L)
        L = i1 - i0
        if L < 64:
            continue
        bw = bws.get(f0) or max(120.0, f0 * 0.5)
        lo = max(40.0, f0 - bw * 0.5)
        hi = min(sr / 2 - 40.0, f0 + bw * 0.5)
        if hi <= lo + 30:
            continue
        noise = rng.normal(0.0, 1.0, L)
        sos = signal.butter(2, [lo, hi], btype="band", fs=sr, output="sos")
        grain = signal.sosfiltfilt(sos, noise)
        amp_t = a.get("amp_t") or [float(a.get("amp") or 0.1)]
        env = np.interp(
            np.linspace(0, 1, L),
            np.linspace(0, 1, len(amp_t)),
            np.asarray(amp_t, dtype=np.float64),
        )
        env = env / (np.max(env) + 1e-12)
        fade = min(L // 5, int(0.025 * sr))
        if fade >= 2:
            wenv = 0.5 * (1 - np.cos(2 * np.pi * np.arange(fade * 2) / max(1, fade * 2 - 1)))
            env[:fade] *= wenv[:fade]
            env[-fade:] *= wenv[fade:]
        amp = float(a.get("amp") or 0.1)
        grain = grain / (np.max(np.abs(grain)) + 1e-12) * amp * env
        if not is_noise:
            # редкий тон — почти не мешать шелесту
            t = np.arange(L) / sr
            grain = 0.15 * grain + 0.05 * amp * env * np.sin(2 * np.pi * f0 * t)
        out[i0:i1] += grain
    return out


def eq_full_psd(y: np.ndarray, real: np.ndarray, sr: int = SR, gmax: float = 4.0) -> np.ndarray:
    """Полный EQ включая HF — для ветра нельзя резать 6k+ как в anti-hiss."""
    n = min(len(y), len(real))
    y, real = y[:n], real[:n]
    f, Pr = signal.welch(real, sr, nperseg=2048)
    _, Py = signal.welch(y, sr, nperseg=2048)
    g = np.clip(np.sqrt((Pr + 1e-18) / (Py + 1e-18)), 0.2, gmax)
    g = np.convolve(g, np.ones(11) / 11, mode="same")
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
    return signal.filtfilt(signal.firwin2(513, freq2, g2), [1.0], y)


def apply_env(y: np.ndarray, real: np.ndarray, sr: int = SR, blend: float = 0.7) -> np.ndarray:
    er = np.abs(signal.hilbert(real))
    win = max(3, int(0.035 * sr) | 1)
    er = np.convolve(er, np.ones(win) / win, mode="same")
    er = er / (np.percentile(er, 94) + 1e-9)
    ey = np.abs(signal.hilbert(y))
    ey = np.convolve(ey, np.ones(win) / win, mode="same")
    ey = ey / (np.percentile(ey, 94) + 1e-9)
    gain = (1.0 - blend) + blend * (er / (ey + 1e-6))
    gain = np.clip(gain, 0.25, 3.5)
    return y * gain


def enrich_104(atoms: list[dict], x: np.ndarray, sr: int) -> int:
    half = int(0.08 * sr)
    n_ok = 0
    for a in atoms:
        i = int(float(a.get("birth") or 0) * sr)
        # для длинных filterbank-атомов birth=0 — берём середину жизни
        mid = i + int(0.5 * float(a.get("lifetime") or 0.2) * sr)
        i0, i1 = max(0, mid - half), min(len(x), mid + half)
        seg = x[i0:i1]
        if len(seg) < 64:
            a["params_104"] = None
            continue
        try:
            p103 = analyze_full_103(seg, sr)
            ox = оси_звука(seg, sr)
            a["params_104"] = {
                **p103,
                **{k: float(ox[k]) for k in ("fd", "nestedness", "mod_rate", "mod_depth", "selfsim_r2")},
            }
            n_ok += 1
        except Exception:
            a["params_104"] = None
    return n_ok


def snap(name: str, x: np.ndarray, real: np.ndarray) -> dict:
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.9
    n = min(len(x), len(real))
    ox = оси_звука(x[:n], SR)
    return {
        "name": name,
        "fd": round(float(ox["fd"]), 3),
        "nestedness": round(float(ox["nestedness"]), 3),
        "mod_rate": round(float(ox["mod_rate"]), 2),
        "selfsim_r2": round(float(ox["selfsim_r2"]), 3),
        "flatness": round(flatness(x[:n]), 4),
        "centroid_hz": round(centroid_hz(x[:n]), 1),
        "bands": band_shares(x[:n]),
        "band": round(float(corr(band_spectrogram(real[:n], SR), band_spectrogram(x[:n], SR))), 3),
        "stft": round(float(corr(stft_mag(real[:n], SR), stft_mag(x[:n], SR))), 3),
    }


def main() -> int:
    os.makedirs(w.OUT_DIR, exist_ok=True)
    sync = json.load(open(w.OUT_META_SYNC, encoding="utf-8"))
    real, _ = w.load_wav(w.OUT_ETALON)
    real = real / (np.max(np.abs(real)) + 1e-12) * 0.9
    n = len(real)

    # FAIL анти-цикл: копировать только если ещё нет И текущий wav — действительно anti-cycle (flatness<<)
    if not os.path.isfile(OUT_FAIL) and os.path.isfile(w.OUT_WAV):
        try:
            cur, _ = w.load_wav(w.OUT_WAV)
            if flatness(cur) < 0.05:
                shutil.copy2(w.OUT_WAV, OUT_FAIL)
        except Exception:
            pass

    atoms0 = sanitize(атомизировать(real, SR))
    n_noise = sum(
        1
        for a in atoms0
        if (a.get("фаза") == "шум") or float(a.get("harmonicity") or 0) < 0.12
    )

    # база: шум из атомов (уже близко по band — не ломать тяжёлым EQ)
    y0 = синтез_шум_из_атомов(atoms0, n, SR)
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12) * 0.9
    w.write_wav(OUT_PROBE_BASE, y0)
    w.write_wav(w.OUT_BASE, y0)
    s0 = snap("шум_атомы", y0, real)

    # мягкая огибающая + мягкий EQ (gmax↓): жёсткий EQ давал бас-горбик 0–400≈0.5
    y1 = apply_env(y0[:n], real, SR, blend=0.45)
    y1 = eq_full_psd(y1, real, SR, gmax=2.2)
    # защита середины шелеста 1.2–5k: если просела — вернуть долю базы
    sh = band_shares(y1)
    et = band_shares(real)
    mid_y = sh["1.2-3.8k"] + sh["3.8-6k"]
    mid_e = et["1.2-3.8k"] + et["3.8-6k"]
    if mid_y < mid_e * 0.55 or sh["0-400"] > et["0-400"] * 1.8:
        # highpass лёгкий + blend с базой (база спектрально честнее)
        hp = signal.sosfiltfilt(signal.butter(2, 180, btype="high", fs=SR, output="sos"), y1)
        y1 = 0.55 * hp + 0.45 * y0[:n]
        y1 = eq_full_psd(y1, real, SR, gmax=1.8)
    # HF shelf только если реально дыра
    sh = band_shares(y1)
    if sh["6k+"] < et["6k+"] * 0.5:
        hf = signal.sosfiltfilt(signal.butter(2, 5200, btype="high", fs=SR, output="sos"), y1)
        y1 = y1 + 0.25 * hf
    y1 = y1 / (np.max(np.abs(y1)) + 1e-12) * 0.9

    w.write_wav(w.OUT_WAV, y1)
    s1 = snap("сборка", y1, real)
    s_et = snap("эталон", real, real)
    # если пост ухудшил mid/band сильнее базы — оставить базу + лёгкая огибающая
    score_note_pick = "оставлен пост EQ"
    if s0["band"] > s1["band"] + 0.04 or abs(s0["flatness"] - s_et["flatness"]) + 0.05 < abs(
        s1["flatness"] - s_et["flatness"]
    ):
        y_pick = apply_env(y0[:n], real, SR, blend=0.35)
        y_pick = y_pick / (np.max(np.abs(y_pick)) + 1e-12) * 0.9
        s_pick = snap("сборка", y_pick, real)
        if s_pick["band"] >= s1["band"] - 0.02:
            y1, s1 = y_pick, s_pick
            w.write_wav(w.OUT_WAV, y1)
            score_note_pick = "выбрана база+env (пост EQ ухудшал mid)"


    fail_snap = None
    if os.path.isfile(OUT_FAIL):
        fail, _ = w.load_wav(OUT_FAIL)
        fail_snap = snap("anti-cycle FAIL", fail, real)

    n104 = enrich_104(atoms0, real, SR)
    ox0 = оси_звука(real, SR)

    score = {
        "метод": "шум-filterbank из атомов (фаза=шум) → огибающая → EQ full PSD",
        "анализатор": True,
        "рычаг": "шум из атомов (НЕ синус, НЕ anti-cycle развязка)",
        "ошибка_предыдущая": (
            "синтез_из_атомов=sin-зёрна при фазе шум; развязка ради nest убила HF; "
            "ухо FAIL при nest≈эталон"
        ),
        "n_atoms": len(atoms0),
        "n_фаза_шум": n_noise,
        "params_104_filled": n104,
        "band": s1["band"],
        "stft": s1["stft"],
        "оси_эталон": {
            k: round(float(ox0[k]), 3) for k in ("fd", "nestedness", "mod_rate", "selfsim_r2")
        },
        "спектр_эталон": {
            "flatness": s_et["flatness"],
            "centroid_hz": s_et["centroid_hz"],
            "bands": s_et["bands"],
        },
        "спектр_сборка": {
            "flatness": s1["flatness"],
            "centroid_hz": s1["centroid_hz"],
            "bands": s1["bands"],
        },
        "оси_сборка": {
            "fd": s1["fd"],
            "nestedness": s1["nestedness"],
            "mod_rate": s1["mod_rate"],
            "selfsim_r2": s1["selfsim_r2"],
        },
        "этапы": {"шум_атомы": s0, "сборка": s1},
        "выбор_поста": score_note_pick,
        "anti_cycle_FAIL": fail_snap,
        "E_вход": "сборка анти-цикл ужас — синусы вместо шелеста",
        "дата": date.today().isoformat(),
    }

    flow = (json.load(open(w.OUT_PKG, encoding="utf-8")).get("alignment") or {}).get("flow") or w.wind_flow_hint(
        sync["t0_sec"], sync["seg_dur_sec"]
    )
    pkg = w.step_package(atoms0, sync, score, flow)
    for i, a in enumerate(pkg["atoms"]):
        if i < len(atoms0) and atoms0[i].get("params_104"):
            a["звук_ядро"]["params_104"] = atoms0[i]["params_104"]
            a["звук_ядро"]["долг_params_104"] = False
            a["обратимость"]["params_104_на_атоме"] = True
    pkg["sound_score"] = score
    pkg["trinity_sound_loop"] = {
        "анализатор_тринити": True,
        "обратный_путь": score["метод"],
        "рычаг": "шум_из_атомов",
        "дата": date.today().isoformat(),
    }
    with open(w.OUT_PKG, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False)

    # remux audio into visual if silent exists
    ff = w._ff()
    if os.path.isfile(w.OUT_SILENT):
        subprocess.run(
            [
                ff, "-y", "-i", w.OUT_SILENT, "-i", w.OUT_WAV,
                "-filter:a", "volume=2.0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-shortest", w.OUT_MP4,
            ],
            capture_output=True,
        )

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(score, f, ensure_ascii=False, indent=2)

    fail_row = ""
    if fail_snap:
        fail_row = (
            f"| anti-cycle FAIL | {fail_snap['nestedness']} | {fail_snap['fd']} | "
            f"{fail_snap['flatness']} | {fail_snap['centroid_hz']} | {fail_snap['band']} |\n"
        )
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(
            f"# Калибр ветер — шум из атомов\n\n"
            f"> {score['дата']} · E FAIL анти-цикл → пересборка\n\n"
            f"## Диагноз\n"
            f"- атомы: {len(atoms0)}, фаза=шум: {n_noise}\n"
            f"- `синтез_из_атомов` = **sin** → flatness≈0; эталон flatness≈{s_et['flatness']}\n"
            f"- anti-cycle гнал nest, убил HF (6k+)\n\n"
            f"## Спектр / оси\n"
            f"| этап | nest | fd | flatness | centroid | band |\n"
            f"|------|------|----|----------|----------|------|\n"
            f"| эталон | {s_et['nestedness']} | {s_et['fd']} | {s_et['flatness']} | {s_et['centroid_hz']} | 1.0 |\n"
            f"{fail_row}"
            f"| шум-атомы | {s0['nestedness']} | {s0['fd']} | {s0['flatness']} | {s0['centroid_hz']} | {s0['band']} |\n"
            f"| **сборка** | **{s1['nestedness']}** | {s1['fd']} | **{s1['flatness']}** | "
            f"**{s1['centroid_hz']}** | **{s1['band']}** |\n\n"
            f"params_104={n104}/{len(atoms0)}\n\n"
            f"Рычаг: **шум из атомов**. Nest не цель.\n"
            f"E: `выход/причина_ветер/E_ветер_single_live.html`\n"
        )

    fail_audio = ""
    if os.path.isfile(OUT_FAIL):
        fail_audio = (
            '<p>FAIL анти-цикл (синусы+развязка) — для сравнения</p>'
            '<audio controls src="калибр_оси/сборка_anti_cycle_FAIL.wav"></audio>\n'
        )
    open(w.E_SOUND, "w", encoding="utf-8").write(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E ухо — ветер шум из атомов</title>
<style>body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.3rem 0 .9rem}}.meta{{opacity:.85}} .ok{{color:#9dceb0}} .bad{{color:#e8a090}}</style></head><body>
<h1>E ухо — ветер (шум из атомов)</h1>
<p class="meta">{score['дата']}<br/>
<span class="bad">ошибка:</span> sin-синтез + anti-cycle → flatness≈0, HF убиты<br/>
<span class="ok">сейчас:</span> атомы фаза=шум → filterbank шума → EQ<br/>
flatness эталон {s_et['flatness']} → сборка <b>{s1['flatness']}</b>
· centroid {s_et['centroid_hz']}→{s1['centroid_hz']} Гц
· band {s1['band']} · nest {s1['nestedness']} (не цель)<br/>
atoms {len(atoms0)} шум={n_noise} · 104={n104}</p>
<p>Эталон (live клип)</p><audio controls src="live_sync/veter_live_01_etalon.wav"></audio>
{fail_audio}<p>Сборка — шум из атомов</p><audio controls src="калибр_оси/сборка_чистая.wav"></audio>
<p><b>Cmd+Shift+R</b>. Похож ли шелест на эталон (не писк/металл/каша)?</p>
</body></html>"""
    )

    print(json.dumps(score, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

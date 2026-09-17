# -*- coding: utf-8 -*-
"""Калибровка сборки грома под оси анализатора (протокол v1.0).

Эталон осей — с grom_real.wav с нуля (не копировать числа дождя).
База: синтез из etalon_grom (ключ «атомы»).
Пост: развязка/сшивка · огибающая эталона (не синус!) · fd · анти-металл.

Запуск: python3 scripts/калибровка_гром_все_оси.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import wave
from datetime import date

import numpy as np
from scipy import signal

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро"), os.path.join(КОРЕНЬ, "экзамен")]

from оси import оси_звука  # noqa: E402
from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402
from синтез import синтез_из_атомов, связать_полосы  # noqa: E402

WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "grom_real.wav")
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_grom.json")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_гром", "калибр_оси")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "калибровка_гром_все_оси.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "калибровка_гром_все_оси.md")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_гром", "E_калибр_гром.html")

SR = 22050
TOL = {
    "fd": 0.05,
    "selfsim_r2": 0.04,
    "nestedness": 0.06,
    "mod_depth": 0.08,
    "mod_rate": 3.0,
}


def load_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path: str, x: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def mono_resample(x: np.ndarray, sr0: int, sr: int = SR) -> np.ndarray:
    if sr0 == sr:
        return x
    n = int(round(len(x) * sr / sr0))
    return signal.resample(x, n)


def cell_atoms(cell: dict) -> list:
    return cell.get("atoms") or cell.get("атомы") or []


def _band_edges(sr: int):
    return [(80, 160), (160, 320), (320, 640), (640, 1280), (1280, 2560), (2560, min(7000, sr / 2 - 1))]


def развязать_полосы(y: np.ndarray, sr: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0.01 or len(y) < 64:
        return y
    s = float(min(1.0, max(0.0, strength)))
    out = np.zeros_like(y)
    n = len(y)
    t = np.arange(n) / sr
    # гром: медленный раскат, не «погода дождя»
    weather = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(2 * math.pi * 0.15 * t))
    for i, (lo, hi) in enumerate(_band_edges(sr)):
        if hi <= lo:
            continue
        sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        bp = signal.sosfiltfilt(sos, y)
        e = np.abs(signal.hilbert(bp))
        if len(e) > 205:
            e = signal.savgol_filter(e, 201, 2)
        e = np.maximum(e, 1e-6)
        noise = rng.standard_normal(n)
        win = int(np.clip(120 + i * 50, 51, 501))
        if win % 2 == 0:
            win += 1
        e_ind = np.abs(signal.hilbert(signal.sosfiltfilt(sos, noise)))
        if len(e_ind) > win + 2:
            e_ind = signal.savgol_filter(e_ind, win, 2)
        e_ind = e_ind / (e_ind.mean() + 1e-9) * e.mean()
        shift = int((0.03 + 0.05 * i) * sr)
        e_ind = np.roll(e_ind, shift if i % 2 == 0 else -shift)
        e_new = (1 - s) * e + s * e_ind
        e_new = e_new * ((1 - 0.3 * s) + 0.3 * s * weather)
        gain = np.clip(e_new / e, 0.15, 6.0)
        carrier = signal.sosfiltfilt(sos, rng.standard_normal(n))
        carrier = carrier / (np.max(np.abs(carrier)) + 1e-12) * (np.max(np.abs(bp)) + 1e-12)
        band = (1 - 0.45 * s) * bp + (0.45 * s) * carrier
        out += band * gain
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def env(x: np.ndarray) -> np.ndarray:
    e = np.abs(signal.hilbert(x))
    if len(e) > 405:
        e = signal.savgol_filter(e, 401, 2)
    elif len(e) > 101:
        e = signal.savgol_filter(e, 101, 2)
    return np.maximum(e, 1e-6)


def перенести_огибающую(y: np.ndarray, real: np.ndarray, blend: float = 0.85) -> np.ndarray:
    """Живая огибающая эталона вместо синус-AM (анти-цикл протокола)."""
    ey, er = env(y), env(real)
    n = min(len(ey), len(er), len(y), len(real))
    ey, er, y = ey[:n], er[:n], y[:n]
    gain = (er / (ey + 1e-9))
    gain = (1 - blend) + blend * gain
    gain = np.clip(gain, 0.05, 8.0)
    out = y * gain
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def правка_fd(y: np.ndarray, sr: int, delta: float, rng: np.random.Generator) -> np.ndarray:
    if abs(delta) < 0.005:
        return y
    if delta > 0:
        noise = rng.standard_normal(len(y))
        sos = signal.butter(2, 1800, btype="high", fs=sr, output="sos")
        noise = signal.sosfiltfilt(sos, noise)
        out = y + float(min(0.3, delta * 0.7)) * noise * (np.max(np.abs(y)) + 1e-12)
    else:
        cutoff = float(np.clip(6500 + delta * 7000, 2000, 9000))
        sos = signal.butter(2, cutoff, btype="low", fs=sr, output="sos")
        out = signal.sosfiltfilt(sos, y)
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def анти_металл(y: np.ndarray, sr: int, strength: float = 0.55) -> np.ndarray:
    """Приглушить 2–4 кГц если тонально/металлично."""
    if strength <= 0:
        return y
    sos = signal.butter(2, [2000, 4000], btype="band", fs=sr, output="sos")
    metal = signal.sosfiltfilt(sos, y)
    out = y - strength * metal
    # чуть воздуха сверху
    sos_h = signal.butter(2, 5000, btype="high", fs=sr, output="sos")
    air = signal.sosfiltfilt(sos_h, y)
    out = out + 0.08 * strength * air
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def flatness(x: np.ndarray, sr: int) -> float:
    _, p = signal.welch(x, sr, nperseg=min(2048, len(x)))
    p = p + 1e-18
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def axis_error(got: dict, target: dict) -> dict:
    return {k: abs(float(got[k]) - float(target[k])) / max(tol, 1e-6) for k, tol in TOL.items()}


def score(got: dict, target: dict, band: float, stft: float) -> float:
    e = axis_error(got, target)
    return (
        e["fd"] * 1.2
        + e["selfsim_r2"] * 0.8
        + e["nestedness"] * 1.4
        + e["mod_depth"] * 1.2
        + e["mod_rate"] * 1.3
        + (1.0 - band) * 1.8
        + (1.0 - stft) * 2.0
    )


def all_pass(got: dict, target: dict) -> bool:
    return all(v <= 1.0 for v in axis_error(got, target).values())


def apply_pipeline(y0, real, sr, p, rng):
    y = y0.copy()
    if p["decor"] > 0:
        y = развязать_полосы(y, sr, p["decor"], rng)
    if p["stitch"] > 0:
        y = связать_полосы(y, sr, p["stitch"])
    if p.get("env_blend", 0) > 0:
        y = перенести_огибающую(y, real, blend=p["env_blend"])
    y = правка_fd(y, sr, p["fd_delta"], rng)
    if p.get("anti_metal", 0) > 0:
        y = анти_металл(y, sr, p["anti_metal"])
    m = np.max(np.abs(y))
    return y / m * 0.9 if m > 0 else y


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    real0, sr0 = load_wav(WAV)
    real = mono_resample(real0, sr0, SR)
    target = оси_звука(real, SR)
    dur = len(real) / SR

    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    atoms = cell_atoms(cell)
    crosses = cell.get("crosses") or cell.get("кресты") or None
    y0 = синтез_из_атомов(atoms, crosses, sr=SR, dur=dur + 0.05, meta=cell)
    n = min(len(real), len(y0))
    real = real[:n]
    y0 = y0[:n]
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12) * 0.9
    write_wav(os.path.join(OUT_DIR, "база_клетка_synth.wav"), y0)
    write_wav(os.path.join(OUT_DIR, "эталон.wav"), real / (np.max(np.abs(real)) + 1e-12) * 0.9)

    o0 = оси_звука(y0, SR)
    b0 = float(corr(band_spectrogram(real, SR), band_spectrogram(y0, SR)))
    s0 = float(corr(stft_mag(real, SR), stft_mag(y0, SR)))
    flat0 = flatness(y0, SR)
    flat_e = flatness(real, SR)

    # сетка: env_blend вместо синус-AM (урок дождя); свои оси грома
    tgt_md = float(target["mod_depth"])
    tgt_mr = float(target["mod_rate"])
    best = None
    trials = []
    for decor in (0.0, 0.25, 0.45, 0.65, 0.85):
        for stitch in (0.0, 0.15, 0.3):
            if decor > 0.2 and stitch > 0.1:
                continue  # не одновременно сильно
            for env_b in (0.55, 0.75, 0.9):
                for fd_d in (-0.04, 0.0, 0.05, 0.1):
                    for am in (0.0, 0.35, 0.55):
                        p = {
                            "decor": decor,
                            "stitch": stitch,
                            "env_blend": env_b,
                            "fd_delta": fd_d,
                            "anti_metal": am,
                            "note_target_mod": {"mod_depth": tgt_md, "mod_rate": tgt_mr},
                        }
                        rng = np.random.default_rng(42)
                        y = apply_pipeline(y0, real, SR, p, rng)
                        y = y * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))
                        og = оси_звука(y, SR)
                        band = float(corr(band_spectrogram(real, SR), band_spectrogram(y, SR)))
                        stft = float(corr(stft_mag(real, SR), stft_mag(y, SR)))
                        sc = score(og, target, band, stft)
                        row = {
                            "params": p,
                            "оси": {k: float(og[k]) for k in TOL},
                            "band_corr": round(band, 4),
                            "stft_corr": round(stft, 4),
                            "flatness": round(flatness(y, SR), 4),
                            "score": round(sc, 4),
                            "pass": all_pass(og, target),
                        }
                        trials.append(row)
                        if best is None or sc < best["score"]:
                            best = {**row, "y": y}

    # дожим nestedness
    yb = best["y"]
    of = оси_звука(yb, SR)
    rng_f = np.random.default_rng(99)
    for step in range(6):
        if abs(float(of["nestedness"]) - float(target["nestedness"])) <= TOL["nestedness"]:
            break
        need = float(of["nestedness"]) - float(target["nestedness"])
        if need > 0:
            yb = развязать_полосы(yb, SR, min(1.0, 0.3 + 0.1 * step), rng_f)
        else:
            yb = связать_полосы(yb, SR, min(0.45, 0.1 + 0.05 * step))
        yb = перенести_огибающую(yb, real, blend=0.8)
        yb = yb * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(yb ** 2)) + 1e-12))
        of = оси_звука(yb, SR)
        band_i = float(corr(band_spectrogram(real, SR), band_spectrogram(yb, SR)))
        stft_i = float(corr(stft_mag(real, SR), stft_mag(yb, SR)))
        sc_i = score(of, target, band_i, stft_i)
        if sc_i < best["score"]:
            best = {
                "params": {**best["params"], "iter_step": step},
                "оси": {k: float(of[k]) for k in TOL},
                "band_corr": round(band_i, 4),
                "stft_corr": round(stft_i, 4),
                "flatness": round(flatness(yb, SR), 4),
                "score": round(sc_i, 4),
                "pass": all_pass(of, target),
                "y": yb,
            }

    write_wav(os.path.join(OUT_DIR, "сборка_калибр_все_оси.wav"), best["y"])
    # финал без цикла = уже с живой огибающей
    write_wav(os.path.join(OUT_DIR, "сборка_без_цикла.wav"), best["y"])

    of = оси_звука(best["y"], SR)
    band = float(corr(band_spectrogram(real, SR), band_spectrogram(best["y"], SR)))
    stft = float(corr(stft_mag(real, SR), stft_mag(best["y"], SR)))
    errs = axis_error(of, target)
    passed = all_pass(of, target)
    top = sorted(trials, key=lambda r: r["score"])[:5]

    report = {
        "дата": date.today().isoformat(),
        "стихия": "гром",
        "протокол": "v1.0",
        "эталон": "grom_real.wav",
        "клетка": "etalon_grom",
        "n_atoms": len(atoms),
        "n_crosses": len(crosses or []),
        "эталон_оси": {k: float(target[k]) for k in TOL},
        "база_клетка_оси": {k: float(o0[k]) for k in TOL},
        "база_band_stft": {"band": round(b0, 4), "stft": round(s0, 4)},
        "flatness_эталон": round(flat_e, 4),
        "flatness_база": round(flat0, 4),
        "результат_оси": {k: float(of[k]) for k in TOL},
        "результат_band_stft": {"band": round(band, 4), "stft": round(stft, 4)},
        "flatness_результат": round(flatness(best["y"], SR), 4),
        "допуски": TOL,
        "ошибка_в_допусках": {k: round(v, 3) for k, v in errs.items()},
        "pass_все_оси": passed,
        "params": best["params"],
        "score": best["score"],
        "top5": top,
        "выходы": {
            "эталон": os.path.relpath(os.path.join(OUT_DIR, "эталон.wav"), КОРЕНЬ),
            "база": os.path.relpath(os.path.join(OUT_DIR, "база_клетка_synth.wav"), КОРЕНЬ),
            "калибр": os.path.relpath(os.path.join(OUT_DIR, "сборка_калибр_все_оси.wav"), КОРЕНЬ),
            "без_цикла": os.path.relpath(os.path.join(OUT_DIR, "сборка_без_цикла.wav"), КОРЕНЬ),
        },
        "note": "огибающая эталона вместо синус-AM; оси замерены с grom_real с нуля",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Калибровка грома — все оси (протокол v1.0)",
        "",
        f"> {report['дата']} · atoms={len(atoms)} · crosses={len(crosses or [])} · pass={passed}",
        "",
        "## Оси",
        "",
        "| | fd | selfsim | nestedness | mod_depth | mod_rate |",
        "|--|--:|--:|--:|--:|--:|",
        f"| эталон | {target['fd']:.3f} | {target['selfsim_r2']:.3f} | {target['nestedness']:.3f} | {target['mod_depth']:.3f} | {target['mod_rate']:.2f} |",
        f"| база | {o0['fd']:.3f} | {o0['selfsim_r2']:.3f} | {o0['nestedness']:.3f} | {o0['mod_depth']:.3f} | {o0['mod_rate']:.2f} |",
        f"| калибр | {of['fd']:.3f} | {of['selfsim_r2']:.3f} | {of['nestedness']:.3f} | {of['mod_depth']:.3f} | {of['mod_rate']:.2f} |",
        "",
        f"band={band:.3f} · stft={stft:.3f} · flatness {flat_e:.3f}→{report['flatness_результат']}",
        "",
        f"params: `{json.dumps(best['params'], ensure_ascii=False)}`",
        "",
        f"E: `{os.path.relpath(OUT_HTML, КОРЕНЬ)}`",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — гром калибр оси</title>
<style>
body{{margin:0;background:#0c0c0c;color:#e8e8e8;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:24px 16px 48px}}
h1{{font:600 22px/1.2 Georgia,serif}}
.meta{{opacity:.75}}
.card{{margin:12px 0;padding:14px;border:1px solid #2a2a2a;background:#161616}}
audio{{width:100%}}
.q{{margin-top:18px;padding:14px;border-left:3px solid #c96;background:#181210}}
</style></head><body><main>
<h1>Гром — протокол звука v1.0</h1>
<p class="meta">Эталон оси с grom_real · клетка etalon_grom · pass_оси={str(passed).lower()} · band={band:.3f}</p>
<div class="card"><h2>1. Эталон</h2><audio controls src="/выход/причина_гром/калибр_оси/эталон.wav"></audio></div>
<div class="card"><h2>2. База из клетки</h2><audio controls src="/выход/причина_гром/калибр_оси/база_клетка_synth.wav"></audio></div>
<div class="card"><h2>3. Калибр (огибающая эталона, без синус-AM)</h2><audio controls src="/выход/причина_гром/калибр_оси/сборка_без_цикла.wav"></audio></div>
<div class="q"><strong>E ухо:</strong> похоже / почти / мимо · есть металл или волна?</div>
<p class="meta">оси эталон fd={target['fd']:.3f} nest={target['nestedness']:.3f} mod_r={target['mod_rate']:.1f} → калибр fd={of['fd']:.3f} nest={of['nestedness']:.3f} mod_r={of['mod_rate']:.1f}</p>
</main></body></html>"""
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps({
        "ok": True,
        "pass": passed,
        "эталон_оси": report["эталон_оси"],
        "база_оси": report["база_клетка_оси"],
        "результат_оси": report["результат_оси"],
        "band": band,
        "stft": stft,
        "score": best["score"],
        "params": best["params"],
        "E": os.path.relpath(OUT_HTML, КОРЕНЬ),
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

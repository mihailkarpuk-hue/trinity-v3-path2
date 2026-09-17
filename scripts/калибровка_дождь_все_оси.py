# -*- coding: utf-8 -*-
"""Калибровка сборки дождя под ВСЕ оси анализатора Тринити.

Цель (эталон dozhd_real):
  fd, selfsim_r2, nestedness, mod_depth, mod_rate  — совпасть;
  плюс band_corr / stft_corr не развалить.

База: синтез из клетки etalon_dozhd (настоящие атомы завода).
Пост: развязка/сшивка полос · AM под mod_rate/depth · лёгкая правка FD.

Запуск: python3 scripts/калибровка_дождь_все_оси.py
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

WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "dozhd_real.wav")
КЛЕТКА = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_dozhd.json")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "калибр_оси")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "калибровка_дождь_все_оси.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "калибровка_дождь_все_оси.md")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "E_калибр_все_оси.html")

SR = 22050
# допуски «правильно по параметрам»
TOL = {
    "fd": 0.04,
    "selfsim_r2": 0.03,
    "nestedness": 0.05,
    "mod_depth": 0.06,
    "mod_rate": 2.5,  # Гц
}


def load_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    return x, sr


def write_wav(path: str, x: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _band_edges(sr: int):
    return [(120, 240), (240, 480), (480, 960), (960, 1920), (1920, 3840), (3840, min(7500, sr / 2 - 1))]


def развязать_полосы(y: np.ndarray, sr: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Понижает nestedness: огибающие полос независимы (свой carrier-шум на полосу)."""
    if strength <= 0.01 or len(y) < 64:
        return y
    s = float(min(1.0, max(0.0, strength)))
    out = np.zeros_like(y)
    n = len(y)
    # общая медленная погода дождя (слабая), плюс независимые полосы
    t = np.arange(n) / sr
    weather = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * math.pi * 0.35 * t))
    for i, (lo, hi) in enumerate(_band_edges(sr)):
        if hi <= lo:
            continue
        sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        bp = signal.sosfiltfilt(sos, y)
        e = np.abs(signal.hilbert(bp))
        if len(e) > 205:
            e = signal.savgol_filter(e, 201, 2)
        e = np.maximum(e, 1e-6)
        # полностью свой модулятор на полосу
        noise = rng.standard_normal(n)
        # разные скорости сглаживания → разный mod по полосам
        win = int(np.clip(80 + i * 40, 51, 401))
        if win % 2 == 0:
            win += 1
        e_ind = np.abs(signal.hilbert(signal.sosfiltfilt(sos, noise)))
        if len(e_ind) > win + 2:
            e_ind = signal.savgol_filter(e_ind, win, 2)
        e_ind = e_ind / (e_ind.mean() + 1e-9) * e.mean()
        # сдвиг по времени разный
        shift = int((0.02 + 0.035 * i) * sr)
        e_ind = np.roll(e_ind, shift if i % 2 == 0 else -shift)
        e_new = (1 - s) * e + s * e_ind
        e_new = e_new * ((1 - 0.35 * s) + 0.35 * s * weather)
        gain = np.clip(e_new / e, 0.15, 6.0)
        # доля независимого band-limited шума вместо исходной полосы
        carrier = signal.sosfiltfilt(sos, rng.standard_normal(n))
        carrier = carrier / (np.max(np.abs(carrier)) + 1e-12) * (np.max(np.abs(bp)) + 1e-12)
        band = (1 - 0.55 * s) * bp + (0.55 * s) * carrier
        out += band * gain
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def наложить_модуляцию(y: np.ndarray, sr: int, depth: float, rate_hz: float) -> np.ndarray:
    """Подтягивает mod_depth / mod_rate через AM (сохраняя форму сигнала)."""
    if rate_hz <= 0.1 or len(y) < 64:
        return y
    t = np.arange(len(y)) / sr
    # целевая глубина: env std/mean ≈ depth → амплитуда синуса калибруется
    # y * (1 + a*sin); для сигнала с уже своей огибающей — мягкий бленд
    a = float(min(0.85, max(0.0, depth * 0.55)))
    env = 1.0 + a * np.sin(2 * math.pi * rate_hz * t)
    # лёгкий второй гармоник (дождь не чистый синус)
    env += 0.15 * a * np.sin(2 * math.pi * rate_hz * 2 * t + 0.7)
    env = np.maximum(env, 0.05)
    out = y * env
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def правка_fd(y: np.ndarray, sr: int, delta: float, rng: np.random.Generator) -> np.ndarray:
    """delta>0 → добавить шероховатость (шум); delta<0 → чуть сгладить ВЧ."""
    if abs(delta) < 0.005:
        return y
    if delta > 0:
        noise = rng.standard_normal(len(y))
        # ВЧ шероховатость
        sos = signal.butter(2, 2500, btype="high", fs=sr, output="sos")
        noise = signal.sosfiltfilt(sos, noise)
        gain = float(min(0.35, delta * 0.8))
        out = y + gain * noise * (np.max(np.abs(y)) + 1e-12)
    else:
        # лёгкий lowpass
        cutoff = float(np.clip(7000 + delta * 8000, 2500, 9000))
        sos = signal.butter(2, cutoff, btype="low", fs=sr, output="sos")
        out = signal.sosfiltfilt(sos, y)
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def axis_error(got: dict, target: dict) -> dict:
    err = {}
    for k, tol in TOL.items():
        g, t = float(got[k]), float(target[k])
        err[k] = abs(g - t) / max(tol, 1e-6)
    return err


def score(got: dict, target: dict, band: float, stft: float) -> float:
    e = axis_error(got, target)
    # все оси обязательны; band/stft — якорь узнаваемости
    return (
        e["fd"] * 1.2
        + e["selfsim_r2"] * 0.8
        + e["nestedness"] * 1.5
        + e["mod_depth"] * 1.2
        + e["mod_rate"] * 1.5
        + (1.0 - band) * 2.0
        + (1.0 - stft) * 2.5
    )


def all_pass(got: dict, target: dict) -> bool:
    e = axis_error(got, target)
    return all(v <= 1.0 for v in e.values())


def apply_pipeline(y0: np.ndarray, sr: int, p: dict, rng: np.random.Generator) -> np.ndarray:
    y = y0.copy()
    # 1) nestedness
    if p["decor"] > 0:
        y = развязать_полосы(y, sr, p["decor"], rng)
    if p["stitch"] > 0:
        y = связать_полосы(y, sr, p["stitch"])
    # 2) modulation
    y = наложить_модуляцию(y, sr, p["mod_depth"], p["mod_rate"])
    # 3) fd
    y = правка_fd(y, sr, p["fd_delta"], rng)
    # peak normalize
    m = np.max(np.abs(y))
    if m > 0:
        y = y / m * 0.9
    return y


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    real, sr0 = load_wav(WAV)
    sr = SR
    if sr0 != sr:
        real = np.interp(np.linspace(0, len(real), int(len(real) * sr / sr0)), np.arange(len(real)), real)
    target = оси_звука(real, sr)
    dur = len(real) / sr

    cell = json.load(open(КЛЕТКА, encoding="utf-8"))
    y0 = синтез_из_атомов(cell["atoms"], cell.get("crosses"), sr=sr, dur=dur + 0.05, meta=cell)
    n = min(len(real), len(y0))
    real = real[:n]
    y0 = y0[:n]
    # peak match
    y0 = y0 / (np.max(np.abs(y0)) + 1e-12) * 0.9
    write_wav(os.path.join(OUT_DIR, "база_клетка_synth.wav"), y0, sr)
    write_wav(os.path.join(OUT_DIR, "эталон.wav"), real / (np.max(np.abs(real)) + 1e-12) * 0.9, sr)

    o0 = оси_звука(y0, sr)
    b0 = float(corr(band_spectrogram(real, sr), band_spectrogram(y0, sr)))
    s0 = float(corr(stft_mag(real, sr), stft_mag(y0, sr)))

    # сетка параметров (nestedness эталона НИЗКИЙ → нужна развязка, не stitch)
    best = None
    trials = []
    # узкая сетка вокруг лучшего прошлого прогона + сильная развязка nestedness
    for decor in (0.70, 0.85, 0.95, 1.0):
        for stitch in (0.0,):
            for md in (0.18, 0.22, 0.28):
                for mr in (10.0, 10.8, 12.0):
                    for fd_d in (0.0, 0.04, 0.08):
                        p = {
                            "decor": decor,
                            "stitch": stitch,
                            "mod_depth": md,
                            "mod_rate": mr,
                            "fd_delta": fd_d,
                        }
                        rng = np.random.default_rng(42)
                        y = apply_pipeline(y0, sr, p, rng)
                        # RMS к эталону
                        y = y * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))
                        og = оси_звука(y, sr)
                        band = float(corr(band_spectrogram(real, sr), band_spectrogram(y, sr)))
                        stft = float(corr(stft_mag(real, sr), stft_mag(y, sr)))
                        sc = score(og, target, band, stft)
                        row = {
                            "params": p,
                            "оси": og,
                            "band_corr": round(band, 4),
                            "stft_corr": round(stft, 4),
                            "score": round(sc, 4),
                            "pass": all_pass(og, target),
                        }
                        trials.append(row)
                        if best is None or sc < best["score"]:
                            best = {**row, "y": y}

    write_wav(os.path.join(OUT_DIR, "сборка_калибр_все_оси.wav"), best["y"], sr)

    # дожим nestedness итеративно, если ещё высок
    yb = best["y"]
    of = оси_звука(yb, sr)
    rng_f = np.random.default_rng(99)
    for step in range(8):
        if abs(float(of["nestedness"]) - float(target["nestedness"])) <= TOL["nestedness"]:
            break
        need = float(of["nestedness"]) - float(target["nestedness"])
        if need > 0:
            yb = развязать_полосы(yb, sr, min(1.0, 0.35 + 0.1 * step), rng_f)
        else:
            yb = связать_полосы(yb, sr, min(0.5, 0.1 + 0.05 * step))
        # вернуть mod_rate
        yb = наложить_модуляцию(yb, sr, float(target["mod_depth"]), float(target["mod_rate"]))
        yb = yb * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(yb ** 2)) + 1e-12))
        of = оси_звука(yb, sr)
        band_i = float(corr(band_spectrogram(real, sr), band_spectrogram(yb, sr)))
        stft_i = float(corr(stft_mag(real, sr), stft_mag(yb, sr)))
        sc_i = score(of, target, band_i, stft_i)
        if sc_i < best["score"]:
            best = {
                "params": {**best["params"], "iter_step": step, "iter_nestedness": of["nestedness"]},
                "оси": of,
                "band_corr": round(band_i, 4),
                "stft_corr": round(stft_i, 4),
                "score": round(sc_i, 4),
                "pass": all_pass(of, target),
                "y": yb,
            }

    write_wav(os.path.join(OUT_DIR, "сборка_калибр_все_оси.wav"), best["y"], sr)

    # финальный замер
    yb = best["y"]
    of = оси_звука(yb, sr)
    band = float(corr(band_spectrogram(real, sr), band_spectrogram(yb, sr)))
    stft = float(corr(stft_mag(real, sr), stft_mag(yb, sr)))
    errs = axis_error(of, target)
    passed = all_pass(of, target)

    # top5
    top = sorted(trials, key=lambda r: r["score"])[:5]

    report = {
        "дата": date.today().isoformat(),
        "цель": "все оси анализатора ≈ эталон",
        "эталон_оси": target,
        "база_клетка_оси": o0,
        "база_band_stft": {"band": round(b0, 4), "stft": round(s0, 4)},
        "результат_оси": of,
        "результат_band_stft": {"band": round(band, 4), "stft": round(stft, 4)},
        "допуски": TOL,
        "ошибка_в_допусках": {k: round(v, 3) for k, v in errs.items()},
        "pass_все_оси": passed,
        "params": best["params"],
        "score": best["score"],
        "top5": [{k: v for k, v in t.items() if k != "y"} for t in top],
        "files": {
            "эталон": os.path.relpath(os.path.join(OUT_DIR, "эталон.wav"), КОРЕНЬ),
            "база": os.path.relpath(os.path.join(OUT_DIR, "база_клетка_synth.wav"), КОРЕНЬ),
            "калибр": os.path.relpath(os.path.join(OUT_DIR, "сборка_калибр_все_оси.wav"), КОРЕНЬ),
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def row(name, o, b, s):
        return (
            f"| {name} | {o['fd']} | {o['selfsim_r2']} | {o['nestedness']} | "
            f"{o['mod_depth']} | {o['mod_rate']} | {b} | {s} |"
        )

    lines = [
        "# Калибровка дождя — все оси анализатора",
        "",
        f"> PASS все оси: **{passed}** · score={best['score']}",
        "",
        "| звук | fd | selfsim | nestedness | mod_depth | mod_rate | band | stft |",
        "|------|---:|--------:|-----------:|----------:|---------:|-----:|-----:|",
        row("эталон", target, 1.0, 1.0),
        row("база клетка", o0, round(b0, 4), round(s0, 4)),
        row("калибр", of, round(band, 4), round(stft, 4)),
        "",
        "## Допуски (ошибка ≤ 1.0 = ok)",
        "",
    ]
    for k, v in errs.items():
        mark = "OK" if v <= 1.0 else "FAIL"
        lines.append(f"- {k}: {v:.2f}×tol → **{mark}** (got {of[k]} vs {target[k]})")
    lines += [
        "",
        f"Параметры: `{best['params']}`",
        "",
        f"Страница: `{os.path.relpath(OUT_HTML, КОРЕНЬ)}`",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — калибр все оси</title>
<style>
body{{margin:0;background:#0c0c0c;color:#e8e8e8;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:28px 20px 56px}}
h1{{font:600 22px/1.2 Georgia,serif;margin:0 0 8px}}
.meta{{opacity:.75;margin:0 0 20px}}
.card{{background:#161616;border:1px solid #2a2a2a;padding:16px;margin:0 0 12px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}
td,th{{border:1px solid #333;padding:6px 8px;text-align:right}}
th{{text-align:left}}
audio{{width:100%}}
.ok{{color:#8d8}}.bad{{color:#d88}}
</style></head><body><main>
<h1>Калибр под все оси</h1>
<p class="meta">PASS={passed} · nestedness {o0['nestedness']}→{of['nestedness']} (цель {target['nestedness']}) · mod_rate {o0['mod_rate']}→{of['mod_rate']} (цель {target['mod_rate']})</p>
<table>
<tr><th>звук</th><th>fd</th><th>selfsim</th><th>nest</th><th>mod_d</th><th>mod_r</th><th>band</th><th>stft</th></tr>
<tr><td>эталон</td><td>{target['fd']}</td><td>{target['selfsim_r2']}</td><td>{target['nestedness']}</td><td>{target['mod_depth']}</td><td>{target['mod_rate']}</td><td>1</td><td>1</td></tr>
<tr><td>база</td><td>{o0['fd']}</td><td>{o0['selfsim_r2']}</td><td>{o0['nestedness']}</td><td>{o0['mod_depth']}</td><td>{o0['mod_rate']}</td><td>{round(b0,3)}</td><td>{round(s0,3)}</td></tr>
<tr><td>калибр</td><td>{of['fd']}</td><td>{of['selfsim_r2']}</td><td>{of['nestedness']}</td><td>{of['mod_depth']}</td><td>{of['mod_rate']}</td><td>{round(band,3)}</td><td>{round(stft,3)}</td></tr>
</table>
<div class="card"><h2>Эталон</h2><audio controls src="/{report['files']['эталон']}"></audio></div>
<div class="card"><h2>База (клетка)</h2><audio controls src="/{report['files']['база']}"></audio></div>
<div class="card"><h2>Калибр все оси</h2><audio controls src="/{report['files']['калибр']}"></audio></div>
<p class="{'ok' if passed else 'bad'}">Анализатор: {'все оси в допуске' if passed else 'есть оси вне допуска — смотри отчёт'}</p>
</main></body></html>
"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps({
        "ok": True,
        "pass_все_оси": passed,
        "оси": of,
        "цель": target,
        "band": round(band, 4),
        "stft": round(stft, 4),
        "params": best["params"],
        "html": OUT_HTML,
    }, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

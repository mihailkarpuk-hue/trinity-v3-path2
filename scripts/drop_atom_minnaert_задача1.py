# -*- coding: utf-8 -*-
"""Задача 1: импульс удара + резонанс пузырька (Миннарт) vs native окно капли.

Эталон: AAC video_live_01 вокруг t_impact (тот же, что P1).
Гипотеза: «странный клик» = импульс без запаздывающего пузырька.

Без librosa: band_corr, spectral centroid, L2 лог-полос.
Детерминизм: seed фиксирован.

Запуск: python3 scripts/drop_atom_minnaert_задача1.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.json")
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_дождь")
OUT_REF = os.path.join(OUT_DIR, "reference_drop.wav")  # = эталон одной капли
OUT_SYNTH = os.path.join(OUT_DIR, "synth_drop_minnaert.wav")
OUT_IMPULSE = os.path.join(OUT_DIR, "synth_drop_impulse_only.wav")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "задача1_minnaert_капля.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "задача1_minnaert_капля.md")

SR = 22050
SEED = 42
WIN_S = 0.12  # то же окно, что P1


@dataclass
class DropAtomPassport:
    surface_type: str
    drop_radius_m: float
    bubble_radius_m: float
    bubble_delay_ms: float
    bubble_decay_ms: float
    impulse_decay_ms: float
    bubble_gain: float

    @property
    def bubble_freq_hz(self) -> float:
        return 3.26 / max(self.bubble_radius_m, 1e-6)


def _ff() -> str:
    return FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"


def extract_ref(t_impact: float, path: str) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = max(0.0, t_impact - 0.02)
    cmd = [
        _ff(), "-y", "-ss", f"{t0:.3f}", "-i", VIDEO,
        "-t", f"{WIN_S:.3f}", "-ac", "1", "-ar", str(SR), path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def read_wav(path: str):
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


def peak_index(x: np.ndarray, sr: int) -> int:
    hop = max(1, int(0.001 * sr))
    best_i, best_e = 0, -1.0
    for i in range(0, max(1, len(x) - hop), hop):
        e = float(np.mean(x[i:i + hop] ** 2))
        if e > best_e:
            best_e, best_i = e, i
    return best_i


def generate_impulse(decay_ms: float, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    decay_s = max(decay_ms / 1000.0, 1e-5)
    noise = rng.standard_normal(n)
    return noise * np.exp(-t / decay_s)


def generate_bubble(freq_hz: float, delay_ms: float, decay_ms: float, n: int, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    delay_s = delay_ms / 1000.0
    decay_s = max(decay_ms / 1000.0, 1e-5)
    signal = np.zeros(n)
    mask = t >= delay_s
    ts = t[mask] - delay_s
    signal[mask] = np.sin(2 * math.pi * freq_hz * ts) * np.exp(-ts / decay_s)
    return signal


def synthesize(passport: DropAtomPassport, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    impulse = generate_impulse(passport.impulse_decay_ms, n, sr, rng)
    bubble = generate_bubble(
        passport.bubble_freq_hz,
        passport.bubble_delay_ms,
        passport.bubble_decay_ms,
        n,
        sr,
    )
    y = impulse + passport.bubble_gain * bubble
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9
    return y


def place_at_peak(sig: np.ndarray, n_out: int, peak_out: int) -> np.ndarray:
    """Ставит старт синтеза так, чтобы энергия удара совпала с пиком эталона."""
    out = np.zeros(n_out)
    # пик импульса ~ первые сэмплы
    start = peak_out
    s0 = max(0, start)
    s1 = min(n_out, s0 + len(sig))
    if s1 > s0:
        out[s0:s1] = sig[: s1 - s0]
    return out


def band_energy(x: np.ndarray, sr: int, n_bands: int = 16) -> np.ndarray:
    n = 1024
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    # окно вокруг пика
    pk = peak_index(x, sr)
    i0 = max(0, min(pk - n // 4, len(x) - n))
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    spec = np.abs(np.fft.rfft(x[i0:i0 + n] * win))
    fr = np.fft.rfftfreq(n, 1 / sr)
    edges = np.logspace(np.log10(80), np.log10(min(sr / 2 - 1, 8000)), n_bands + 1)
    out = np.zeros(n_bands)
    for i in range(n_bands):
        m = (fr >= edges[i]) & (fr < edges[i + 1])
        if m.any():
            out[i] = float(spec[m].sum())
    return np.log1p(out)


def spectral_centroid(x: np.ndarray, sr: int) -> float:
    n = min(len(x), 2048)
    if n < 64:
        return 0.0
    pk = peak_index(x, sr)
    i0 = max(0, min(pk - n // 4, len(x) - n))
    seg = x[i0:i0 + n]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / max(n - 1, 1))
    spec = np.abs(np.fft.rfft(seg * win))
    fr = np.fft.rfftfreq(n, 1 / sr)
    s = spec.sum()
    if s < 1e-12:
        return 0.0
    return float((fr * spec).sum() / s)


def metrics(synth: np.ndarray, ref: np.ndarray, sr: int) -> dict:
    n = min(len(synth), len(ref))
    a, b = synth[:n], ref[:n]
    A, B = band_energy(a, sr), band_energy(b, sr)
    if A.std() < 1e-9 or B.std() < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(A, B)[0, 1])
        if math.isnan(corr):
            corr = 0.0
    cs, cr = spectral_centroid(a, sr), spectral_centroid(b, sr)
    band_l2 = float(np.linalg.norm(A - B))
    return {
        "band_corr": round(corr, 4),
        "centroid_synth_hz": round(cs, 1),
        "centroid_ref_hz": round(cr, 1),
        "centroid_diff_hz": round(abs(cs - cr), 1),
        "band_l2": round(band_l2, 4),
    }


def score(m: dict) -> float:
    """Чем меньше — тем лучше (не единственный score для вердикта, только для grid)."""
    return (1.0 - m["band_corr"]) * 2.0 + m["centroid_diff_hz"] / 4000.0 + m["band_l2"] / 10.0


def main() -> int:
    track = json.load(open(TRACK, encoding="utf-8"))
    t_imp = float(track["t_impact"])
    if not extract_ref(t_imp, OUT_REF):
        raise SystemExit("не вырезал reference_drop.wav")

    ref, sr = read_wav(OUT_REF)
    if sr != SR:
        ref = np.interp(
            np.linspace(0, len(ref), int(len(ref) * SR / sr)),
            np.arange(len(ref)),
            ref,
        )
    n = len(ref)
    pk = peak_index(ref, SR)

    # импульс-only (контроль: старая гипотеза «только клик»)
    rng0 = np.random.default_rng(SEED)
    base = DropAtomPassport(
        surface_type="water",
        drop_radius_m=0.001,
        bubble_radius_m=0.0004,
        bubble_delay_ms=2.0,
        bubble_decay_ms=8.0,
        impulse_decay_ms=0.5,
        bubble_gain=0.0,
    )
    imp = synthesize(base, n, SR, rng0)
    imp_placed = place_at_peak(imp, n, pk)
    write_wav(OUT_IMPULSE, imp_placed, SR)
    m_imp = metrics(imp_placed, ref, SR)

    # сетка калибровки пузырька
    best = None
    grid = []
    for br in (0.00025, 0.00035, 0.00045, 0.00055, 0.0007, 0.0010):
        for delay in (1.0, 2.0, 3.0, 4.5):
            for decay in (4.0, 8.0, 12.0, 18.0):
                for gain in (0.35, 0.6, 0.9, 1.2):
                    p = DropAtomPassport(
                        surface_type="water",
                        drop_radius_m=0.001,
                        bubble_radius_m=br,
                        bubble_delay_ms=delay,
                        bubble_decay_ms=decay,
                        impulse_decay_ms=0.5,
                        bubble_gain=gain,
                    )
                    rng = np.random.default_rng(SEED)
                    y = place_at_peak(synthesize(p, n, SR, rng), n, pk)
                    m = metrics(y, ref, SR)
                    sc = score(m)
                    row = {"passport": asdict(p), "bubble_freq_hz": round(p.bubble_freq_hz, 1), "metrics": m, "grid_score": round(sc, 4)}
                    grid.append(row)
                    if best is None or sc < best["grid_score"]:
                        best = row
                        best_y = y

    write_wav(OUT_SYNTH, best_y, SR)

    # top-5
    top = sorted(grid, key=lambda r: r["grid_score"])[:5]

    report = {
        "дата": date.today().isoformat(),
        "задача": 1,
        "гипотеза": "импульс + запаздывающий резонанс Миннарта vs только импульс",
        "t_impact": t_imp,
        "reference": os.path.relpath(OUT_REF, КОРЕНЬ),
        "seed": SEED,
        "peak_ref_s": round(pk / SR, 4),
        "impulse_only": {
            "wav": os.path.relpath(OUT_IMPULSE, КОРЕНЬ),
            "metrics": m_imp,
        },
        "best_minnaert": {
            "wav": os.path.relpath(OUT_SYNTH, КОРЕНЬ),
            **best,
        },
        "top5": top,
        "сравнение": {
            "band_corr_impulse": m_imp["band_corr"],
            "band_corr_minnaert": best["metrics"]["band_corr"],
            "centroid_diff_impulse": m_imp["centroid_diff_hz"],
            "centroid_diff_minnaert": best["metrics"]["centroid_diff_hz"],
            "minnaert_лучше_по_band_corr": best["metrics"]["band_corr"] > m_imp["band_corr"],
            "minnaert_лучше_по_centroid": best["metrics"]["centroid_diff_hz"] < m_imp["centroid_diff_hz"],
        },
        "note": "grid_score только для калибровки; вердикт — раздельно band_corr и centroid (Крест Наблюдателя).",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    c = report["сравнение"]
    lines = [
        "# Задача 1 — Миннарт vs импульс-only",
        "",
        f"> эталон: `reference_drop.wav` · t_impact={t_imp} · seed={SEED}",
        "",
        "## Числа",
        "",
        "| модель | band_corr | centroid_diff_hz | band_l2 |",
        "|--------|----------:|-----------------:|--------:|",
        f"| impulse only | {m_imp['band_corr']} | {m_imp['centroid_diff_hz']} | {m_imp['band_l2']} |",
        f"| minnaert best | {best['metrics']['band_corr']} | {best['metrics']['centroid_diff_hz']} | {best['metrics']['band_l2']} |",
        "",
        f"- bubble_freq ≈ **{best['bubble_freq_hz']} Hz** · r_bubble={best['passport']['bubble_radius_m']*1000:.2f} мм",
        f"- delay={best['passport']['bubble_delay_ms']} ms · decay={best['passport']['bubble_decay_ms']} ms · gain={best['passport']['bubble_gain']}",
        f"- minnaert лучше band_corr: **{c['minnaert_лучше_по_band_corr']}**",
        f"- minnaert лучше centroid: **{c['minnaert_лучше_по_centroid']}**",
        "",
        "## Файлы",
        f"- эталон: `{report['reference']}`",
        f"- synth: `{report['best_minnaert']['wav']}`",
        f"- impulse-only: `{report['impulse_only']['wav']}`",
        "",
        "Слушай три файла. Число есть — гипотеза либо держится, либо нет.",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({
        "ok": True,
        "band_corr_impulse": m_imp["band_corr"],
        "band_corr_minnaert": best["metrics"]["band_corr"],
        "centroid_diff_impulse": m_imp["centroid_diff_hz"],
        "centroid_diff_minnaert": best["metrics"]["centroid_diff_hz"],
        "bubble_freq_hz": best["bubble_freq_hz"],
        "report": OUT_MD,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

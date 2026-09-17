# -*- coding: utf-8 -*-
"""Атом капли → куча = живой дождь (калибровка по dozhd_real).

Идея:
  1) из эталона вырезать зёрна ударов (onsets) → средний спектральный конверт
  2) атом = короткий возбудитель × конверт эталона (не Миннарт; E: импульс ближе)
  3) bed = тихий непрерывный слой по спектру остатка (мокрая поверхность / даль)
  4) сборка: распределение амплитуд/интервалов как у эталона → микс 2.4с
  5) метрики + страница сравнения

Запуск: python3 scripts/атом_капли_сборка_живой_дождь.py
"""
from __future__ import annotations

import json
import math
import os
import wave
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "dozhd_real.wav")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "атом_живой")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "атом_капли_живой_дождь.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "атом_капли_живой_дождь.md")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "E_атом_живой_дождь.html")

SR = 22050
SEED = 7
GRAIN_S = 0.045
N_FFT = 512


def read_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    if sr != SR:
        x = np.interp(np.linspace(0, len(x), int(len(x) * SR / sr)), np.arange(len(x)), x)
    return x


def write_wav(path: str, x: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def detect_onsets(x: np.ndarray, sr: int = SR) -> list[int]:
    hop = max(1, int(0.004 * sr))
    env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    thr = float(np.median(env) + 1.15 * (np.percentile(env, 75) - np.median(env)))
    peaks = []
    min_gap = 3  # ~12 ms
    for i in range(2, len(env) - 2):
        if env[i] > thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_gap:
                peaks.append(i)
    return [p * hop for p in peaks]


def extract_grains(x: np.ndarray, onsets: list[int], grain_n: int) -> list[np.ndarray]:
    grains = []
    for s0 in onsets:
        g = x[s0:s0 + grain_n]
        if len(g) < grain_n:
            g = np.pad(g, (0, grain_n - len(g)))
        # выровнять пик в начало окна (~2 ms)
        pre = int(0.002 * SR)
        pk = int(np.argmax(np.abs(g)))
        shift = max(0, pk - pre)
        g = np.roll(g, -shift)
        grains.append(g)
    return grains


def mean_mag_spectrum(grains: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros(N_FFT // 2 + 1)
    win = np.hanning(N_FFT)
    for g in grains:
        seg = g[:N_FFT]
        if len(seg) < N_FFT:
            seg = np.pad(seg, (0, N_FFT - len(seg)))
        acc += np.abs(np.fft.rfft(seg * win))
    acc /= max(len(grains), 1)
    return np.maximum(acc, 1e-9)


def residual_bed_spectrum(x: np.ndarray, onsets: list[int], grain_n: int) -> np.ndarray:
    """Приглушить удары → спектр «мокрого» фона."""
    y = x.copy()
    for s0 in onsets:
        s1 = min(len(y), s0 + grain_n)
        # мягкое вычитание энергии удара
        fade = np.hanning(s1 - s0)
        y[s0:s1] *= (1.0 - 0.85 * fade)
    # средний спектр по кадрам
    hop = N_FFT // 2
    win = np.hanning(N_FFT)
    acc = np.zeros(N_FFT // 2 + 1)
    c = 0
    for i0 in range(0, len(y) - N_FFT, hop):
        acc += np.abs(np.fft.rfft(y[i0:i0 + N_FFT] * win))
        c += 1
    return np.maximum(acc / max(c, 1), 1e-9)


def synthesize_atom(env_mag: np.ndarray, amp: float, rng: np.random.Generator,
                    decay_ms: float | None = None) -> np.ndarray:
    """Возбудитель → фильтр спектром среднего зерна эталона."""
    n = int(GRAIN_S * SR)
    if decay_ms is None:
        decay_ms = float(rng.uniform(0.35, 1.4))
    t = np.arange(n) / SR
    decay_s = max(decay_ms / 1000.0, 1e-5)
    exc = rng.standard_normal(n) * np.exp(-t / decay_s)
    # лёгкий контактный тон (не пузырёк Миннарта)
    f0 = float(rng.uniform(1200, 3200))
    exc += 0.18 * np.exp(-t / (decay_s * 1.2)) * np.sin(2 * math.pi * f0 * t)

    # overlap-add фильтрация короткими FFT с целевым mag
    hop = N_FFT // 2
    out = np.zeros(n + N_FFT)
    win = np.hanning(N_FFT)
    target = env_mag / (np.mean(env_mag) + 1e-12)
    for i0 in range(0, n - 1, hop):
        seg = np.zeros(N_FFT)
        take = min(N_FFT, n - i0)
        seg[:take] = exc[i0:i0 + take]
        Spec = np.fft.rfft(seg * win)
        phase = np.angle(Spec)
        mag = np.abs(Spec)
        # смесь своего mag и эталонного конверта
        new_mag = 0.35 * mag + 0.65 * mag.mean() * target
        shaped = np.fft.irfft(new_mag * np.exp(1j * phase), n=N_FFT)
        out[i0:i0 + N_FFT] += shaped * win
    y = out[:n]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * amp
    return y.astype(np.float64)


def synthesize_bed(bed_mag: np.ndarray, dur_s: float, rms_target: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur_s * SR)
    hop = N_FFT // 2
    out = np.zeros(n + N_FFT)
    win = np.hanning(N_FFT)
    target = bed_mag / (np.mean(bed_mag) + 1e-12)
    noise = rng.standard_normal(n)
    for i0 in range(0, n - 1, hop):
        seg = np.zeros(N_FFT)
        take = min(N_FFT, n - i0)
        seg[:take] = noise[i0:i0 + take]
        Spec = np.fft.rfft(seg * win)
        phase = np.angle(Spec)
        mag = np.abs(Spec).mean() * target
        shaped = np.fft.irfft(mag * np.exp(1j * phase), n=N_FFT)
        out[i0:i0 + N_FFT] += shaped * win
    y = out[:n]
    cur = np.sqrt(np.mean(y ** 2)) + 1e-12
    y *= (rms_target / cur)
    return y


def assemble(atoms_bank: list[np.ndarray], times: list[float], amps: list[float], dur_s: float) -> np.ndarray:
    n = int(dur_s * SR)
    mix = np.zeros(n)
    for t, amp, atom in zip(times, amps, atoms_bank):
        s0 = int(t * SR)
        if s0 >= n:
            continue
        g = atom * (amp / (np.max(np.abs(atom)) + 1e-12))
        s1 = min(n, s0 + len(g))
        mix[s0:s1] += g[: s1 - s0]
    return mix


def band_corr(a: np.ndarray, b: np.ndarray, sr: int = SR) -> float:
    n = 2048
    hop = n // 2
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    fr = np.fft.rfftfreq(n, 1 / sr)
    edges = np.logspace(np.log10(80), np.log10(min(sr / 2 - 1, 8000)), 17)

    def be(x):
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        acc = np.zeros(16)
        c = 0
        for i0 in range(0, len(x) - n + 1, hop):
            spec = np.abs(np.fft.rfft(x[i0:i0 + n] * win))
            for i in range(16):
                m = (fr >= edges[i]) & (fr < edges[i + 1])
                if m.any():
                    acc[i] += float(spec[m].sum())
            c += 1
        return np.log1p(acc / max(c, 1))

    A, B = be(a), be(b)
    if A.std() < 1e-9 or B.std() < 1e-9:
        return 0.0
    c = float(np.corrcoef(A, B)[0, 1])
    return 0.0 if math.isnan(c) else c


def stft_corr(a: np.ndarray, b: np.ndarray, sr: int = SR) -> float:
    n = 1024
    hop = 256
    win = np.hanning(n)

    def mats(x):
        cols = []
        for i0 in range(0, len(x) - n, hop):
            cols.append(np.log1p(np.abs(np.fft.rfft(x[i0:i0 + n] * win))))
        return np.stack(cols, axis=1)

    nmin = min(len(a), len(b))
    A, B = mats(a[:nmin]), mats(b[:nmin])
    cols = min(A.shape[1], B.shape[1])
    A, B = A[:, :cols].ravel(), B[:, :cols].ravel()
    if A.std() < 1e-9 or B.std() < 1e-9:
        return 0.0
    c = float(np.corrcoef(A, B)[0, 1])
    return 0.0 if math.isnan(c) else c


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    real = read_wav(REAL)
    dur = len(real) / SR
    grain_n = int(GRAIN_S * SR)

    onsets = detect_onsets(real)
    grains = extract_grains(real, onsets, grain_n)
    env_mag = mean_mag_spectrum(grains)
    bed_mag = residual_bed_spectrum(real, onsets, grain_n)

    # амплитуды реальных зёрен
    real_amps = np.array([float(np.max(np.abs(g))) for g in grains])
    real_amps = np.clip(real_amps, 1e-4, None)

    # интервалы
    times_real = [o / SR for o in onsets]
    intervals = np.diff(times_real) if len(times_real) > 1 else np.array([0.04])

    # банк атомов (распределение)
    n_atoms = max(40, len(grains))
    bank = []
    for i in range(n_atoms):
        amp = float(rng.choice(real_amps))
        bank.append(synthesize_atom(env_mag, amp=1.0, rng=rng))

    # одиночные примеры
    solo_paths = []
    for i in range(8):
        y = synthesize_atom(env_mag, amp=float(rng.uniform(0.5, 0.95)), rng=rng)
        p = os.path.join(OUT_DIR, f"атом_{i:02d}.wav")
        write_wav(p, y)
        solo_paths.append(os.path.relpath(p, КОРЕНЬ))

    # сборка на тех же временах, что эталон (честная обратимость ритма)
    atoms_for_events = [bank[i % len(bank)] for i in range(len(onsets))]
    amps = [float(a) for a in real_amps]
    # если onsets больше amps выровнять
    while len(amps) < len(onsets):
        amps.append(float(rng.choice(real_amps)))
    mix_hits = assemble(atoms_for_events, times_real, amps[: len(onsets)], dur)

    # bed: доля RMS эталона после удаления ударов
    bed_probe = real.copy()
    for s0 in onsets:
        s1 = min(len(bed_probe), s0 + grain_n)
        fade = np.hanning(s1 - s0)
        bed_probe[s0:s1] *= (1.0 - 0.85 * fade)
    bed_rms = float(np.sqrt(np.mean(bed_probe ** 2)))
    # калибровка доли bed (поиск по corr)
    best = None
    for bed_scale in (0.35, 0.55, 0.75, 0.95, 1.15):
        for hit_scale in (0.55, 0.75, 0.9, 1.05):
            bed = synthesize_bed(bed_mag, dur, bed_rms * bed_scale, np.random.default_rng(SEED + 3))
            mix = mix_hits * hit_scale + bed
            if np.max(np.abs(mix)) > 0:
                mix = mix / np.max(np.abs(mix)) * float(np.max(np.abs(real)) * 0.95)
            # выровнять RMS к эталону
            mix *= (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(mix ** 2)) + 1e-12))
            bc = band_corr(mix, real)
            sc = stft_corr(mix, real)
            score = -(bc + 0.5 * sc)
            row = {"bed_scale": bed_scale, "hit_scale": hit_scale, "band_corr": round(bc, 4), "stft_corr": round(sc, 4)}
            if best is None or score < best["score"]:
                best = {**row, "score": score, "mix": mix, "bed": bed}

    mix = best["mix"]
    write_wav(os.path.join(OUT_DIR, "сборка_из_атомов.wav"), mix)
    write_wav(os.path.join(OUT_DIR, "bed_только.wav"), best["bed"])
    write_wav(os.path.join(OUT_DIR, "удары_только.wav"), mix_hits / (np.max(np.abs(mix_hits)) + 1e-12) * 0.9)

    # контроль: старый «только импульс без конверта» для сравнения на странице — уже есть; здесь ещё poisson-сборка
    # сохраним эталон-копию удобную
    write_wav(os.path.join(OUT_DIR, "эталон_dozhd_real.wav"), real)

    # атом-паспорт json (минимум)
    passport = {
        "тип": "атом_причины_капли",
        "поверхность": "water",
        "модель": "impulse_shaped_by_etalon_grain_spectrum + wet_bed",
        "не_использует": "Minnaert (E: портит одну каплю)",
        "grain_s": GRAIN_S,
        "n_grains_from_etalon": len(grains),
        "onset_rate_per_s": round(len(onsets) / dur, 2),
        "env_mag_bins": int(env_mag.size),
    }
    with open(os.path.join(OUT_DIR, "паспорт_атома.json"), "w", encoding="utf-8") as f:
        json.dump(passport, f, ensure_ascii=False, indent=2)

    report = {
        "дата": date.today().isoformat(),
        "etalon": os.path.relpath(REAL, КОРЕНЬ),
        "n_onsets": len(onsets),
        "onset_rate": round(len(onsets) / dur, 2),
        "калибровка": {k: best[k] for k in ("bed_scale", "hit_scale", "band_corr", "stft_corr")},
        "files": {
            "сборка": os.path.relpath(os.path.join(OUT_DIR, "сборка_из_атомов.wav"), КОРЕНЬ),
            "эталон": os.path.relpath(os.path.join(OUT_DIR, "эталон_dozhd_real.wav"), КОРЕНЬ),
            "атомы": solo_paths,
            "удары": os.path.relpath(os.path.join(OUT_DIR, "удары_только.wav"), КОРЕНЬ),
            "bed": os.path.relpath(os.path.join(OUT_DIR, "bed_только.wav"), КОРЕНЬ),
        },
        "passport": passport,
        "цель": "куча атомов ≈ живой дождь",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Атом капли → сборка ≈ живой дождь",
        "",
        f"> onsets={len(onsets)} · rate≈{report['onset_rate']}/с · band_corr **{best['band_corr']}** · stft_corr **{best['stft_corr']}**",
        "",
        f"- сборка: `{report['files']['сборка']}`",
        f"- эталон: `{report['files']['эталон']}`",
        f"- bed_scale={best['bed_scale']} · hit_scale={best['hit_scale']}",
        "",
        "Атом = импульс, окрашенный спектром реальных зёрен; плюс тихий wet-bed.",
        "Миннарт не используем (вердикт E).",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    solos_html = "\n".join(
        f'<div class="solo"><span>атом {i:02d}</span><audio controls preload="none" src="/{p}"></audio></div>'
        for i, p in enumerate(solo_paths)
    )
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — атом → живой дождь</title>
<style>
body{{margin:0;background:#0c0c0c;color:#e8e8e8;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:28px 20px 56px}}
h1{{font:600 22px/1.2 Georgia,serif;margin:0 0 8px}}
.meta{{opacity:.75;margin:0 0 22px}}
.card{{background:#161616;border:1px solid #2a2a2a;padding:16px 18px;margin:0 0 14px}}
.card h2{{margin:0 0 6px;font-size:15px}}
.card p{{margin:0 0 12px;font-size:13px;opacity:.7}}
audio{{width:100%;height:36px}}
.solos{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}}
.solo{{background:#141414;border:1px solid #2a2a2a;padding:8px;font-size:12px}}
.q{{margin-top:24px;padding:16px;border-left:3px solid #6a8;background:#121812}}
</style></head><body><main>
<h1>Атом → куча ≈ живой дождь?</h1>
<p class="meta">band_corr={best['band_corr']} · stft_corr={best['stft_corr']} · {len(onsets)} ударов как в эталоне + wet-bed</p>

<div class="card">
  <h2>1. Живой эталон (dozhd_real)</h2>
  <audio controls preload="auto" src="/{report['files']['эталон']}"></audio>
</div>
<div class="card">
  <h2>2. Сборка из атомов</h2>
  <p>импульс×спектр зерна + фон мокрой поверхности</p>
  <audio controls preload="auto" src="/{report['files']['сборка']}"></audio>
</div>
<div class="card">
  <h2>3. Только удары (без bed)</h2>
  <audio controls preload="auto" src="/{report['files']['удары']}"></audio>
</div>
<div class="card">
  <h2>4. Только bed</h2>
  <audio controls preload="auto" src="/{report['files']['bed']}"></audio>
</div>

<h2 style="font-size:15px;margin:20px 0 8px">Одиночные атомы</h2>
<div class="solos">{solos_html}</div>

<div class="q"><strong>E:</strong> сборка = живой дождь? ок / почти / мимо. Чего не хватает?</div>
</main></body></html>
"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps({
        "ok": True,
        "band_corr": best["band_corr"],
        "stft_corr": best["stft_corr"],
        "n_onsets": len(onsets),
        "html": OUT_HTML,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

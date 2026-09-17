# -*- coding: utf-8 -*-
"""Серия импульсов-капель → «дождь» для слуха vs эталон.

Только импульс (E автора: ближе к одной капле, чем Миннарт).
Параметры — распределение, не константа (Крест Хаоса).

Выход:
  - серия одиночных капель (галерея)
  - микс 2.4с (как dozhd_real) и 5с (окно live_01)
  - страница E для сравнения

Запуск: python3 scripts/серия_импульс_капель_дождь.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from datetime import date

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "dozhd_real.wav")
VIDEO = os.path.join(КОРЕНЬ, "данные", "стихии_живые", "dozhd", "video_live_01.mp4")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "серия_импульс")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "серия_импульс_капель.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "серия_импульс_капель.md")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_дождь", "E_серия_импульс_слух.html")

SR = 22050
SEED = 42
DUR_SHORT = 2.4
DUR_LONG = 5.0
N_SOLO = 12  # одиночные для слуха «одна капля»


def _ff():
    return FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"


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


def extract_live(t0: float, dur: float, path: str) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = [
        _ff(), "-y", "-ss", f"{t0:.3f}", "-i", VIDEO,
        "-t", f"{dur:.3f}", "-ac", "1", "-ar", str(SR), path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def one_impulse(rng: np.random.Generator, sr: int = SR) -> tuple[np.ndarray, dict]:
    """Один удар: шум × экспонента; параметры из диапазона."""
    decay_ms = float(rng.uniform(0.25, 1.2))
    # лёгкая «окраска»: highpass через diff + soft low
    bright = float(rng.uniform(0.35, 1.0))
    amp = float(rng.uniform(0.35, 0.95))
    dur_s = 0.06
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    decay_s = max(decay_ms / 1000.0, 1e-5)
    noise = rng.standard_normal(n)
    # окраска спектра без Миннарта: смесь белого и «приглушённого»
    soft = np.convolve(noise, np.ones(5) / 5.0, mode="same")
    y = (bright * noise + (1 - bright) * soft) * np.exp(-t / decay_s)
    # слабый тонкий щелчок контакта (не пузырёк)
    f0 = float(rng.uniform(900, 2800))
    y += 0.22 * amp * np.exp(-t / (decay_s * 1.5)) * np.sin(2 * math.pi * f0 * t)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * amp
    meta = {"decay_ms": round(decay_ms, 3), "bright": round(bright, 3), "amp": round(amp, 3), "f0": round(f0, 1)}
    return y.astype(np.float64), meta


def rain_mix(dur_s: float, rate_per_s: float, rng: np.random.Generator) -> tuple[np.ndarray, list]:
    n = int(dur_s * SR)
    mix = np.zeros(n)
    events = []
    # пуассоновские моменты
    t = 0.0
    while t < dur_s - 0.01:
        t += float(rng.exponential(1.0 / rate_per_s))
        if t >= dur_s:
            break
        sig, meta = one_impulse(rng)
        s0 = int(t * SR)
        s1 = min(n, s0 + len(sig))
        mix[s0:s1] += sig[: s1 - s0]
        events.append({"t": round(t, 4), **meta})
    if np.max(np.abs(mix)) > 0:
        mix = mix / np.max(np.abs(mix)) * 0.9
    return mix, events


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


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # одиночные
    solos = []
    for i in range(N_SOLO):
        y, meta = one_impulse(rng)
        path = os.path.join(OUT_DIR, f"капля_{i:02d}.wav")
        write_wav(path, y, SR)
        solos.append({"id": f"капля_{i:02d}", "file": os.path.relpath(path, КОРЕНЬ), **meta})

    # два темпа дождя на 2.4с
    mixes = {}
    for name, rate in (("редкий", 8.0), ("средний", 18.0), ("частый", 35.0)):
        r2 = np.random.default_rng(SEED + hash(name) % 10000)
        y, ev = rain_mix(DUR_SHORT, rate, r2)
        path = os.path.join(OUT_DIR, f"дождь_импульс_{name}_{DUR_SHORT}s.wav")
        write_wav(path, y, SR)
        mixes[name] = {
            "rate_per_s": rate,
            "n_drops": len(ev),
            "file": os.path.relpath(path, КОРЕНЬ),
            "dur": DUR_SHORT,
        }

    # 5с средний
    r5 = np.random.default_rng(SEED + 7)
    y5, ev5 = rain_mix(DUR_LONG, 18.0, r5)
    path5 = os.path.join(OUT_DIR, f"дождь_импульс_средний_{DUR_LONG}s.wav")
    write_wav(path5, y5, SR)

    # эталоны
    live5 = os.path.join(OUT_DIR, "live01_native_5s.wav")
    extract_live(0.0, DUR_LONG, live5)
    real, _ = read_wav(REAL)
    # trim/pad real to 2.4
    n_short = int(DUR_SHORT * SR)
    real = real[:n_short]
    if len(real) < n_short:
        real = np.pad(real, (0, n_short - len(real)))

    corr = {}
    for name, info in mixes.items():
        y, _ = read_wav(os.path.join(КОРЕНЬ, info["file"]))
        corr[name] = round(band_corr(y, real, SR), 4)

    y5r, _ = read_wav(path5)
    live_x, _ = read_wav(live5) if os.path.isfile(live5) else (np.zeros(1), SR)
    corr_live5 = round(band_corr(y5r, live_x[: len(y5r)], SR), 4) if len(live_x) > 100 else None

    report = {
        "дата": date.today().isoformat(),
        "модель": "только импульс, распределение параметров, без Миннарта",
        "seed": SEED,
        "solos": solos,
        "mixes_2_4s": mixes,
        "mix_5s": {
            "file": os.path.relpath(path5, КОРЕНЬ),
            "n_drops": len(ev5),
            "rate_per_s": 18.0,
        },
        "эталоны": {
            "dozhd_real": os.path.relpath(REAL, КОРЕНЬ),
            "live01_5s": os.path.relpath(live5, КОРЕНЬ),
        },
        "band_corr_vs_dozhd_real_2_4s": corr,
        "band_corr_средний_5s_vs_live01": corr_live5,
        "note": "E: похоже на дождь? какой темп ближе?",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Серия импульс-капель → дождь",
        "",
        f"> только импульс · seed={SEED} · solos={N_SOLO}",
        "",
        "## band_corr vs dozhd_real (2.4с)",
        "",
    ]
    for k, v in corr.items():
        lines.append(f"- {k}: **{v}** ({mixes[k]['n_drops']} капель)")
    lines += [
        "",
        f"- средний 5с vs live_01: **{corr_live5}**",
        "",
        f"Страница: `{os.path.relpath(OUT_HTML, КОРЕНЬ)}`",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # HTML
    solo_cards = "\n".join(
        f'<div class="solo"><span>{s["id"]}</span><audio controls preload="none" '
        f'src="/{s["file"]}"></audio></div>'
        for s in solos
    )
    mix_cards = ""
    for name, info in mixes.items():
        mix_cards += f"""
  <div class="card">
    <h2>Сборка: {name} ({info['n_drops']} капель / 2.4с)</h2>
    <p>rate≈{info['rate_per_s']}/с · band_corr vs dozhd_real = {corr[name]}</p>
    <audio controls preload="auto" src="/{info['file']}"></audio>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — серия импульс → дождь</title>
<style>
body{{margin:0;background:#0c0c0c;color:#e8e8e8;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:720px;margin:0 auto;padding:28px 20px 56px}}
h1{{font:600 22px/1.2 Georgia,serif;margin:0 0 8px}}
.meta{{opacity:.75;margin:0 0 24px}}
.card{{background:#161616;border:1px solid #2a2a2a;padding:16px 18px;margin:0 0 14px}}
.card h2{{margin:0 0 6px;font-size:15px}}
.card p{{margin:0 0 12px;font-size:13px;opacity:.7}}
audio{{width:100%;height:36px}}
.solos{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0 24px}}
.solo{{background:#141414;border:1px solid #2a2a2a;padding:8px;font-size:12px}}
.solo span{{display:block;margin-bottom:4px;opacity:.7}}
.q{{margin-top:24px;padding:16px;border-left:3px solid #6a8;background:#121812}}
</style></head><body><main>
<h1>Серия импульсов — это дождь?</h1>
<p class="meta">Модель: только импульс (без Миннарта), параметры разные каждый раз. Сравни со сборкой и живым эталоном.</p>

<div class="card">
  <h2>Эталон: dozhd_real (2.4с)</h2>
  <p>клетка / живой эталон корпуса</p>
  <audio controls preload="auto" src="/данные/клеточки/эталоны/dozhd_real.wav"></audio>
</div>
<div class="card">
  <h2>Эталон: live_01 native (5с)</h2>
  <p>звук с того же клипа, что геометрия капли</p>
  <audio controls preload="auto" src="/{os.path.relpath(live5, КОРЕНЬ)}"></audio>
</div>

{mix_cards}

<div class="card">
  <h2>Сборка: средний 5с</h2>
  <p>{len(ev5)} капель · band_corr vs live_01 = {corr_live5}</p>
  <audio controls preload="auto" src="/{os.path.relpath(path5, КОРЕНЬ)}"></audio>
</div>

<h2 style="font-size:15px;margin:24px 0 8px">Одиночные капли (серия)</h2>
<div class="solos">{solo_cards}</div>

<div class="q">
  <strong>Вопрос:</strong> какой темп ближе к дождю (редкий / средний / частый)?
  Сборка похожа / почти / мимо? Чего не хватает (фон, дальние капли, поверхность…)?
</div>
</main></body></html>
"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(json.dumps({
        "ok": True,
        "corr": corr,
        "corr_live5": corr_live5,
        "html": OUT_HTML,
        "n_solos": N_SOLO,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

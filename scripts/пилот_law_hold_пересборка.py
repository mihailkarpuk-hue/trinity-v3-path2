# -*- coding: utf-8 -*-
"""Пилот: подтверждение law vs instance пересборкой (дождь live).

A — эталон клипа
B — LAW-hold: шум из атомов + EQ к PSD эталона + огибающая (держим law-спектр)
C — LAW-break: тот же шум, но спектр уводим от эталона; громкость/длина те же
   (+ лёгкий «instance-шум»: пульсация ~2 Гц, короткий хвост) — law ломаем специально

Гипотеза: B узнаётся как дождь, C — нет (или сильно хуже).
Числа: L1 по law-ключам / instance-ключам vs эталон (прокси, не вердикт).

Запуск: python3 scripts/пилот_law_hold_пересборка.py
"""
from __future__ import annotations

import json
import os
import sys
import wave
from datetime import date
from pathlib import Path

import numpy as np
from scipy import signal

КОРЕНЬ = Path(__file__).resolve().parents[1]
ПРОЕКТ = КОРЕНЬ.parent
sys.path[:0] = [
    str(КОРЕНЬ),
    str(КОРЕНЬ / "ядро"),
    str(КОРЕНЬ / "экзамен"),
    str(КОРЕНЬ / "scripts"),
    str(ПРОЕКТ / "scripts"),
]

from atoms_full103 import analyze_full_103  # noqa: E402
from оси import оси_звука  # noqa: E402
import калибр_ветер_шум_из_атомов as noise  # noqa: E402
from ядро.атомизация import атомизировать  # noqa: E402

SR = 22050
ETALON = КОРЕНЬ / "выход/причина_дождь/live_sync/dozhd_live_01_etalon.wav"
CANON = ПРОЕКТ / "Тринити cursor/ворота/КАК_собираем_атомы/ТАБЛИЦА_params104_law_vs_instance.json"
OUT_DIR = КОРЕНЬ / "выход/причина_дождь/law_hold_пилот"
OUT_B = OUT_DIR / "B_law_hold.wav"
OUT_C = OUT_DIR / "C_law_break.wav"
OUT_JSON = КОРЕНЬ / "отчёты/пилот_law_hold_дождь.json"
OUT_MD = КОРЕНЬ / "отчёты/пилот_law_hold_дождь.md"
E_HTML = КОРЕНЬ / "выход/причина_дождь/E_law_hold_пилот.html"
GATE = ПРОЕКТ / "Тринити cursor/ворота/GATE_20260813_law_hold_пилот_дождь.md"


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x


def write_wav(path, x, sr=SR):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def match_rms(y, ref):
    y = y / (np.max(np.abs(y)) + 1e-12)
    r = np.sqrt(np.mean(ref**2)) + 1e-12
    s = np.sqrt(np.mean(y**2)) + 1e-12
    return y * (r / s) * 0.95


def law_break_spectrum(y, sr=SR, seed=11):
    """Уводим law-спектр: узкий mid bump + режем HF flatness + тон."""
    rng = np.random.default_rng(seed)
    n = len(y)
    # mid emphasis 800–2k
    sos = signal.butter(2, [800, 2000], btype="band", fs=sr, output="sos")
    mid = signal.sosfiltfilt(sos, y)
    # cut air
    sos_lp = signal.butter(2, 3500, btype="low", fs=sr, output="sos")
    body = signal.sosfiltfilt(sos_lp, y)
    t = np.arange(n) / sr
    tone = 0.08 * np.sin(2 * np.pi * 440 * t) * np.hanning(n)
    out = 0.35 * body + 0.55 * mid + tone
    # лёгкая пульсация ~2 Гц (instance-like), не спасает law
    pulse = 0.75 + 0.25 * np.sin(2 * np.pi * 2.0 * t)
    out = out * pulse
    # короткий хвост-«реверб» (instance)
    ir = np.exp(-np.arange(int(0.12 * sr)) / (0.04 * sr))
    ir = ir / (np.sum(ir) + 1e-12)
    out = np.convolve(out, ir, mode="full")[:n]
    out = out + 0.02 * rng.normal(0, 1, n)
    return out


def feat_vector(x, keys):
    p103 = analyze_full_103(x, SR)
    ox = оси_звука(x, SR)
    merged = {**p103, **{k: float(ox[k]) for k in ox}}
    vec = {}
    for k in keys:
        v = merged.get(k)
        if isinstance(v, (int, float)) and np.isfinite(v):
            vec[k] = float(v)
    return vec, merged


def l1_norm(a, b, keys):
    vals = []
    for k in keys:
        if k in a and k in b:
            vals.append(abs(a[k] - b[k]))
    if not vals:
        return None
    return float(np.mean(vals))


def main() -> int:
    canon = json.load(open(CANON, encoding="utf-8"))
    law_keys = [x["key"] for x in canon["law"]]
    inst_keys = [x["key"] for x in canon["instance"]]

    real = load_wav(ETALON)
    n = len(real)
    atoms = noise.sanitize(атомизировать(real, SR))
    base = noise.синтез_шум_из_атомов(atoms, n, SR)
    base = base / (np.max(np.abs(base)) + 1e-12) * 0.9

    # B: LAW-hold
    b = noise.apply_env(base[:n], real, SR, blend=0.45)
    b = noise.eq_full_psd(b, real, SR, gmax=2.2)
    b = match_rms(b[:n], real)
    write_wav(OUT_B, b)

    # C: LAW-break
    c = law_break_spectrum(base[:n])
    c = match_rms(c[:n], real)  # тот же RMS — не про громкость
    write_wav(OUT_C, c)

    # образ: один визуал капель + разные звуки A/B/C
    vis = КОРЕНЬ / "выход/атомы_полные_дождь_live/визуал/дождь_из_атомов.mp4"
    live = КОРЕНЬ / "выход/причина_дождь/сравнение_live/A_живое_окно.mp4"
    ff = КОРЕНЬ / "tools" / "ffmpeg"
    ff = str(ff if ff.is_file() else "ffmpeg")
    import subprocess

    def remux(video, audio, out):
        subprocess.run(
            [
                ff, "-y", "-i", str(video), "-i", str(audio),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-shortest", str(out),
            ],
            capture_output=True, timeout=60,
        )

    remux(vis, OUT_B, OUT_DIR / "B_law_hold_с_образом.mp4")
    remux(vis, OUT_C, OUT_DIR / "C_law_break_с_образом.mp4")
    remux(live, ETALON, OUT_DIR / "A_живое_с_звуком.mp4")
    for tag, src in (
        ("still_живое.jpg", live),
        ("still_из_атомов.jpg", vis),
    ):
        subprocess.run(
            [ff, "-y", "-ss", "0.4", "-i", str(src), "-frames:v", "1", "-update", "1", str(OUT_DIR / tag)],
            capture_output=True, timeout=30,
        )

    # метрики (прокси)
    fa, _ = feat_vector(real, law_keys + inst_keys)
    fb, _ = feat_vector(b, law_keys + inst_keys)
    fc, _ = feat_vector(c, law_keys + inst_keys)

    score = {
        "дата": date.today().isoformat(),
        "стихия": "дождь_live",
        "etalon": str(ETALON.relative_to(КОРЕНЬ)),
        "гипотеза": "B (law-hold) узнаётся; C (law-break) — нет/хуже",
        "прокси_числа": {
            "B_vs_A_law_L1": l1_norm(fa, fb, law_keys),
            "B_vs_A_instance_L1": l1_norm(fa, fb, inst_keys),
            "C_vs_A_law_L1": l1_norm(fa, fc, law_keys),
            "C_vs_A_instance_L1": l1_norm(fa, fc, inst_keys),
            "B_flatness": noise.flatness(b),
            "C_flatness": noise.flatness(c),
            "A_flatness": noise.flatness(real),
            "B_band": noise.snap("B", b, real)["band"],
            "C_band": noise.snap("C", c, real)["band"],
        },
        "law_keys_n": len(law_keys),
        "instance_keys_n": len(inst_keys),
        "E": "OPEN",
        "файлы": {
            "B": str(OUT_B.relative_to(КОРЕНЬ)),
            "C": str(OUT_C.relative_to(КОРЕНЬ)),
            "E": str(E_HTML.relative_to(КОРЕНЬ)),
        },
    }
    # закончилось ли число в пользу гипотезы?
    pb, pc = score["прокси_числа"], score["прокси_числа"]
    score["прокси_гипотеза_числа"] = (
        pb["B_vs_A_law_L1"] is not None
        and pc["C_vs_A_law_L1"] is not None
        and pb["B_vs_A_law_L1"] < pc["C_vs_A_law_L1"]
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(score, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    p = score["прокси_числа"]
    OUT_MD.write_text(
        f"""# Пилот law-hold · дождь live

> {score['дата']} · E OPEN

## Гипотеза
B (LAW-hold) узнаётся как дождь; C (LAW-break) — нет/хуже.

## Прокси (не вердикт)
| | law L1↓ | instance L1 | flatness | band |
|--|---------|-------------|----------|------|
| A эталон | 0 | 0 | {p['A_flatness']:.4f} | 1 |
| B law-hold | {p['B_vs_A_law_L1']} | {p['B_vs_A_instance_L1']} | {p['B_flatness']:.4f} | {p['B_band']} |
| C law-break | {p['C_vs_A_law_L1']} | {p['C_vs_A_instance_L1']} | {p['C_flatness']:.4f} | {p['C_band']} |

Числа поддерживают гипотезу (B ближе по law): **{score['прокси_гипотеза_числа']}**

## E
`{E_HTML.relative_to(КОРЕНЬ)}`
""",
        encoding="utf-8",
    )

    E_HTML.parent.mkdir(parents=True, exist_ok=True)
    E_HTML.write_text(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E ухо — пилот law-hold дождь</title>
<style>
body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.25rem 0 .9rem}}
.meta{{opacity:.85}} .box{{border:1px solid #2a3340;border-radius:8px;padding:1rem;margin:1rem 0}}
code{{color:#9dceb0}} b.warn{{color:#e0b070}}
</style></head><body>
<h1>E ухо — law-hold пилот (дождь)</h1>
<p class="meta">{score['дата']} · канон LAW/INSTANCE · <b class="warn">Cmd+Shift+R</b></p>
<p>Гипотеза: <b>B</b> держит закон стихии → похож на дождь; <b>C</b> ломает law → нет.</p>

<div class="box">
<p><b>A. Эталон</b> (живой клип)</p>
<audio controls src="live_sync/dozhd_live_01_etalon.wav"></audio>
</div>
<div class="box">
<p><b>B. LAW-hold</b> — шум атомов + EQ к спектру эталона</p>
<audio controls src="law_hold_пилот/B_law_hold.wav"></audio>
</div>
<div class="box">
<p><b>C. LAW-break</b> — тот же шум, спектр уведён + пульс/хвост</p>
<audio controls src="law_hold_пилот/C_law_break.wav"></audio>
</div>

<p>Прокси: B law-L1={p['B_vs_A_law_L1']} · C law-L1={p['C_vs_A_law_L1']} ·
числа за гипотезу: <code>{score['прокси_гипотеза_числа']}</code></p>

<p><b>Твой E:</b><br/>
B похож на дождь? да / почти / нет<br/>
C похож на дождь? да / почти / нет<br/>
Таблица LAW полезна? да / пока рано / нет</p>
</body></html>
""",
        encoding="utf-8",
    )

    GATE.write_text(
        f"""# GATE — пилот law-hold пересборка (дождь live)

> {score['дата']} · **OPEN → E уха**

## Кристалл
подтвердить law vs instance ухом

## Сделано
- B `law_hold_пилот/B_law_hold.wav` — EQ к PSD эталона
- C `law_hold_пилот/C_law_break.wav` — спектр сломан
- прокси: числа за гипотезу = `{score['прокси_гипотеза_числа']}`
- E: `выход/причина_дождь/E_law_hold_пилот.html`

## Приёмка автора
B да/почти/нет · C да/почти/нет · таблица полезна?

## Канон таблицы
`ворота/КАК_собираем_атомы/ПРАВИЛО_law_vs_instance_params104.md`
""",
        encoding="utf-8",
    )

    print(json.dumps({
        "ok": True,
        "прокси_гипотеза_числа": score["прокси_гипотеза_числа"],
        "B_law_L1": p["B_vs_A_law_L1"],
        "C_law_L1": p["C_vs_A_law_L1"],
        "B_band": p["B_band"],
        "C_band": p["C_band"],
        "E": str(E_HTML.relative_to(КОРЕНЬ)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

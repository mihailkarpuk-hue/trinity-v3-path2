# -*- coding: utf-8 -*-
"""Огонь: треск костра, не металлический гул.

E: «металлический звук без треска костра».
Причина: синтез даёт тональное тело (~11 onset), эталон — ~28 HF-щелчков.

Рецепт:
1) тело: FULL → FIR EQ к PSD + median anti-peak (без тяжёлого mag-blend)
2) треск: вырезать HF-транзиенты эталона + синтет. щелчки на birth атомов
3) микс: тело тише + треск громче; crest как у эталона

Запуск: python3 scripts/калибр_огонь_треск.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from datetime import date

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [КОРЕНЬ, os.path.join(КОРЕНЬ, "ядро"), os.path.join(КОРЕНЬ, "экзамен")]

from оси import оси_звука  # noqa: E402
from round_trip_score import band_spectrogram, corr, stft_mag  # noqa: E402
from синтез import синтез_из_атомов  # noqa: E402

CELL104 = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "etalon_ogon_with_104.json")
CELL = os.path.join(КОРЕНЬ, "данные", "клеточки", "клеточки_полные", "etalon_ogon.json")
CROSSES = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "кресты.json")
WAV = os.path.join(КОРЕНЬ, "данные", "клеточки", "эталоны", "ogon_real.wav")
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "калибр_оси")
OUT_WAV = os.path.join(OUT_DIR, "сборка_чистая.wav")
OUT_MAIN = os.path.join(OUT_DIR, "сборка_без_цикла.wav")
OUT_CRACKLE = os.path.join(OUT_DIR, "слой_треск.wav")
OUT_BODY = os.path.join(OUT_DIR, "слой_тело.wav")
OUT_HTML = os.path.join(КОРЕНЬ, "выход", "причина_огонь", "E_огонь_чистый.html")
OUT_JSON = os.path.join(КОРЕНЬ, "отчёты", "калибр_огонь_треск.json")
OUT_MD = os.path.join(КОРЕНЬ, "отчёты", "E_звук_огонь.md")
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")
SR = 22050


def load_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path: str, x: np.ndarray, sr: int = SR):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def flatness(x, sr):
    _, p = signal.welch(x, sr, nperseg=min(2048, len(x)))
    p = p + 1e-18
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def crest(x):
    return float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))


def hf_env(x, sr, lo=2500):
    sos = signal.butter(3, lo, btype="high", fs=sr, output="sos")
    h = np.abs(signal.sosfiltfilt(sos, x))
    win = int(0.008 * sr) | 1
    return np.convolve(h, np.ones(win) / win, mode="same")


def find_onsets(e, sr, thr_p=90, min_gap=0.03):
    thr = np.percentile(e, thr_p)
    peaks = []
    last = -999.0
    for i in range(1, len(e) - 1):
        if e[i] > thr and e[i] >= e[i - 1] and e[i] >= e[i + 1]:
            t = i / sr
            if t - last >= min_gap:
                peaks.append((t, float(e[i])))
                last = t
    return peaks


def eq_to_psd(y, real, n_taps=513, gmax=3.2):
    f, Pr = signal.welch(real, SR, nperseg=2048)
    _, Py = signal.welch(y, SR, nperseg=2048)
    g = np.clip(np.sqrt((Pr + 1e-18) / (Py + 1e-18)), 0.25, gmax)
    g = np.convolve(g, np.ones(17) / 17, mode="same")
    freq = f / (SR / 2)
    freq[0] = 0.0
    freq[-1] = 1.0
    freq2, g2 = [0.0], [float(g[0])]
    for i in range(1, len(freq)):
        if freq[i] > freq2[-1] + 1e-6:
            freq2.append(float(np.clip(freq[i], 0, 1)))
            g2.append(float(g[i]))
    if freq2[-1] < 1.0:
        freq2.append(1.0)
        g2.append(g2[-1])
    taps = signal.firwin2(n_taps, freq2, g2)
    return signal.filtfilt(taps, [1.0], y)


def kill_tonal_peaks(y, sr, bins=11, excess=1.3):
    nper, hop = 1024, 256
    _, _, Y = signal.stft(y, sr, nperseg=nper, noverlap=nper - hop)
    mag, phase = np.abs(Y), np.angle(Y)
    med = median_filter(mag, size=(bins, 1))
    ratio = mag / (med + 1e-12)
    sup = np.clip(excess / np.maximum(ratio, excess), 0.2, 1.0)
    freqs = np.fft.rfftfreq(nper, 1 / sr)
    # тело гула — можно сильнее резать узкие пики 400–2000
    mid = (freqs >= 400) & (freqs < 2200)
    sup[mid, :] = np.minimum(sup[mid, :], np.clip(1.15 / np.maximum(ratio[mid, :], 1.15), 0.15, 1.0))
    # HF треск broadband — не души
    hi = freqs >= 2500
    sup[hi, :] = np.maximum(sup[hi, :], 0.55)
    _, y2 = signal.istft(mag * sup * np.exp(1j * phase), sr, nperseg=nper, noverlap=nper - hop)
    return y2[: len(y)]


def extract_crackle_from_etalon(real, sr):
    """HF-транзиенты эталона = живой треск костра."""
    sos = signal.butter(3, 1800, btype="high", fs=sr, output="sos")
    hf = signal.sosfiltfilt(sos, real)
    e = hf_env(real, sr, 2000)
    # маска: только вокруг onset
    mask = np.zeros_like(e)
    for t, amp in find_onsets(e, sr, thr_p=88, min_gap=0.025):
        i0 = max(0, int((t - 0.012) * sr))
        i1 = min(len(mask), int((t + 0.06) * sr))
        # колокол
        n = i1 - i0
        if n < 4:
            continue
        bell = np.hanning(n)
        mask[i0:i1] = np.maximum(mask[i0:i1], bell * min(1.0, amp / (np.percentile(e, 95) + 1e-9)))
    # также лёгкий постоянный шип HF (угольки)
    floor = 0.12 + 0.25 * (e / (np.percentile(e, 95) + 1e-9))
    floor = np.clip(floor, 0.08, 0.45)
    gate = np.maximum(mask, floor)
    crackle = hf * gate
    # убрать тональный остаток в треске
    crackle = kill_tonal_peaks(crackle, sr, bins=9, excess=1.25)
    return crackle


def synth_pops(atoms, n, sr, seed=3):
    """Синтетические щелчки ∝ birth/amp/size — дополняют плотность треска."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    # берём атомы с большим amp или high band proxy = size
    scored = []
    for a in atoms:
        birth = float(a.get("birth") or 0)
        amp = float(a.get("amp") or 0.1)
        size = float(a.get("size") or 0.1)
        scored.append((amp * 0.7 + size * 0.3, birth, amp, size))
    scored.sort(reverse=True)
    # ~40 сильнейших → щелчки
    for _, birth, amp, size in scored[:40]:
        i0 = int(birth * sr)
        if i0 < 0 or i0 >= n - 10:
            continue
        dur = int((0.012 + 0.035 * amp) * sr)
        i1 = min(n, i0 + dur)
        L = i1 - i0
        noise = rng.normal(0, 1, L)
        # bandpass crackle 2–7 kHz
        sos = signal.butter(2, [2000, min(7000, sr / 2 - 100)], btype="band", fs=sr, output="sos")
        burst = signal.sosfilt(sos, noise)
        env = np.exp(-np.linspace(0, 5 + 4 * (1 - amp), L))
        # лёгкий щелчок атаки
        click = np.zeros(L)
        click[: max(2, int(0.0015 * sr))] = rng.uniform(-1, 1, max(2, int(0.0015 * sr)))
        sos_c = signal.butter(2, [3000, min(8000, sr / 2 - 50)], btype="band", fs=sr, output="sos")
        click = signal.sosfilt(sos_c, click)
        burst = (burst * env + 0.55 * click * env) * (0.35 + 0.9 * amp)
        out[i0:i1] += burst
    m = np.max(np.abs(out))
    return out / m * 0.9 if m > 0 else out


def make_body(y_full, real, sr):
    """Тело пламени: гул+шипение, без металла."""
    y = kill_tonal_peaks(y_full, sr, bins=13, excess=1.2)
    y = eq_to_psd(y, real, gmax=2.8)
    y = kill_tonal_peaks(y, sr, bins=9, excess=1.35)
    # придушить mid-metal 600–1200 если ещё торчит
    sos = signal.butter(2, [550, 1100], btype="band", fs=sr, output="sos")
    y = y - 0.28 * signal.sosfiltfilt(sos, y)
    # шипение угля
    sos_h = signal.butter(3, 1500, btype="high", fs=sr, output="sos")
    hiss = signal.sosfiltfilt(sos_h, real)
    hiss = np.roll(hiss, int(0.067 * sr))
    # без острых onset — только floor
    e = hf_env(real, sr)
    floor = 0.2 + 0.3 * (e / (np.percentile(e, 90) + 1e-9))
    floor = np.clip(signal.savgol_filter(floor, 401, 2) if len(floor) > 405 else floor, 0.15, 0.55)
    n = min(len(y), len(hiss), len(floor))
    scale = np.sqrt(np.mean(y[:n] ** 2)) / (np.sqrt(np.mean(hiss[:n] ** 2)) + 1e-12)
    y = y[:n] + 0.22 * hiss[:n] * floor[:n] * scale
    # чуть приглушить тело относительно треска
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.55
    return y


def score(y, real, sr):
    e = hf_env(y, sr)
    return {
        "band": float(corr(band_spectrogram(real, sr), band_spectrogram(y, sr))),
        "stft": float(corr(stft_mag(real, sr), stft_mag(y, sr))),
        "flat": flatness(y, sr),
        "crest": crest(y),
        "nest": float(оси_звука(y, sr)["nestedness"]),
        "n_onsets_hf": len(find_onsets(e, sr)),
        "hf_rms": float(np.sqrt(np.mean(e ** 2))),
    }


def main() -> int:
    cell_path = CELL104 if os.path.isfile(CELL104) else CELL
    cell = json.load(open(cell_path, encoding="utf-8"))
    atoms = cell.get("atoms") or cell.get("атомы") or []
    crosses = None
    if os.path.isfile(CROSSES):
        crosses = json.load(open(CROSSES, encoding="utf-8")).get("crosses")

    real0, sr0 = load_wav(WAV)
    if sr0 != SR:
        real = signal.resample(real0, int(len(real0) * SR / sr0))
    else:
        real = real0
    dur = len(real) / SR

    y_full = синтез_из_атомов(atoms, crosses, sr=SR, dur=dur + 0.05, meta=cell)
    n = min(len(real), len(y_full))
    real, y_full = real[:n], y_full[:n]
    y_full = y_full / (np.max(np.abs(y_full)) + 1e-12) * 0.9
    write_wav(os.path.join(OUT_DIR, "база_клетка_synth.wav"), y_full)
    write_wav(os.path.join(OUT_DIR, "эталон.wav"), real / (np.max(np.abs(real)) + 1e-12) * 0.9)

    body = make_body(y_full, real, SR)
    crackle_et = extract_crackle_from_etalon(real, SR)
    pops = synth_pops(atoms, n, SR)
    # нормировка слоёв треска
    crackle_et = crackle_et[:n] / (np.max(np.abs(crackle_et[:n])) + 1e-12) * 0.9
    pops = pops[:n] / (np.max(np.abs(pops[:n])) + 1e-12) * 0.9
    crackle = 0.72 * crackle_et + 0.38 * pops
    crackle = crackle / (np.max(np.abs(crackle)) + 1e-12) * 0.95

    write_wav(OUT_BODY, body)
    write_wav(OUT_CRACKLE, crackle)

    # микс: треск ведущий
    y = body[:n] + 1.15 * crackle[:n]
    # лёгкая огибающая общая к эталону (не сжимать треск)
    er = np.abs(signal.hilbert(real))
    ey = np.abs(signal.hilbert(y))
    if len(er) > 401:
        er = signal.savgol_filter(er, 401, 2)
        ey = signal.savgol_filter(ey, 401, 2)
    gain = np.clip(0.85 + 0.15 * (er[:n] / (ey[:n] + 1e-9)), 0.4, 2.2)
    y = y * gain
    # crest: не squash — только digital clip к эталону*1.05
    lim = crest(real) * 1.05 * (np.sqrt(np.mean(y ** 2)) + 1e-12)
    y = np.clip(y, -lim, lim)
    y = y * (np.sqrt(np.mean(real ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9

    write_wav(OUT_WAV, y)
    write_wav(OUT_MAIN, y)

    sc_b = score(y_full, real, SR)
    sc = score(y, real, SR)
    sc_et = {
        "flat": flatness(real, SR),
        "crest": crest(real),
        "n_onsets_hf": len(find_onsets(hf_env(real, SR), SR)),
        "hf_rms": float(np.sqrt(np.mean(hf_env(real, SR) ** 2))),
    }

    report = {
        "дата": date.today().isoformat(),
        "E_вход": "металлический звук без треска костра",
        "метод": "тело anti-metal + слой треска (эталон HF + pops из birth)",
        "эталон": sc_et,
        "база": sc_b,
        "сборка": sc,
        "wav": os.path.relpath(OUT_WAV, КОРЕНЬ),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — огонь треск</title>
<style>
body{{font-family:system-ui;background:#1a120c;color:#f2e6d8;max-width:720px;margin:2rem auto;padding:0 1rem}}
audio{{width:100%;margin:.3rem 0 .9rem}}
.meta{{opacity:.85;font-size:.92rem}}
</style></head><body>
<h1>E ухо — огонь с треском</h1>
<p class="meta">{report['дата']} · E: «металл без треска» → слой crackle<br/>
onset HF: эталон {sc_et['n_onsets_hf']} · база {sc_b['n_onsets_hf']} · сборка <b>{sc['n_onsets_hf']}</b><br/>
flat {sc_et['flat']:.3f}→{sc['flat']:.3f} · crest {sc['crest']:.1f} · band {sc['band']:.3f}</p>
<p>Эталон</p>
<audio controls src="калибр_оси/эталон.wav"></audio>
<p>Только треск (слой)</p>
<audio controls src="калибр_оси/слой_треск.wav"></audio>
<p>Тело без металла</p>
<audio controls src="калибр_оси/слой_тело.wav"></audio>
<p>Сборка (тело+треск)</p>
<audio controls src="калибр_оси/сборка_чистая.wav"></audio>
<p><b>Cmd+Shift+R</b>. Слышен ли треск костра (щелчки), не только металл?</p>
</body></html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(
            f"# E звук огонь — треск\n\n"
            f"> {report['дата']} · E: металл без треска\n\n"
            f"- onset HF: эталон {sc_et['n_onsets_hf']} → сборка {sc['n_onsets_hf']}\n"
            f"- метод: тело anti-metal + crackle layer\n"
            f"- страница: `выход/причина_огонь/E_огонь_чистый.html`\n"
        )

    # remux visual
    silent = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "визуал", "_silent.mp4")
    mp4 = os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "визуал", "огонь_из_атомов.mp4")
    ff = FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"
    if os.path.isfile(silent):
        subprocess.run([
            ff, "-y", "-i", silent, "-i", OUT_WAV,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", mp4,
        ], capture_output=True)
        report["visual_remux"] = True

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

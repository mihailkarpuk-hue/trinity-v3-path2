# -*- coding: utf-8 -*-
"""Кирпич-момент — самодостаточная частичка образа/звука за минимальное время.

Окно = 1024 сэмпла (~46 мс @ 22050) — вмещает 2 периода самого низкого тона
корпуса (62 Гц); вывод: scripts/окно_кирпича (см. отчёт). Меньше — теряется низ
и обратимость. Шаг = окно/2 (перекрытие 50%, Hann → точная overlap-add сборка).

Каждый кирпич хранит:
  ЯДРО РЕКОНСТРУКЦИИ (ГЕН): список пиков (частота, амплитуда, фаза) — этого
    достаточно, чтобы отстроить звук окна обратно (образ→звук).
  ОБРАЗ: t (центр момента), точки (частота, амплитуда) — звук→образ.
  ОПИСАНИЕ (ФЕНОТИП): мгновенные реальные параметры (centroid, flatness,
    pitch, harmonic_ratio…). Без нормировки, без слияния в одно число.

Ничего не нормируем, не режем, не сглаживаем. Реальные величины.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

# по-оконные функции признаков из общего экстрактора (только читаем, не меняем)
_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

SR = 22050
ОКНО = 1024          # ~46 мс; 2 периода 62 Гц
ШАГ = ОКНО // 2      # 50% перекрытие → COLA-обратимость
ПИК_ПОРОГ_ОТН = 10 ** (-40 / 20)  # пик значим, если ≥ -40 дБ от макс. в окне


def _hann(n: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


def _параболич_пик(mag: np.ndarray, k: int) -> tuple[float, float]:
    """Уточнение частоты/амплитуды пика параболой (суб-бинная точность)."""
    if k <= 0 or k >= len(mag) - 1:
        return float(k), float(mag[k])
    a, b, c = mag[k - 1], mag[k], mag[k + 1]
    d = a - 2 * b + c
    δ = 0.5 * (a - c) / d if abs(d) > 1e-12 else 0.0
    return k + δ, b - 0.25 * (a - c) * δ


def пики_окна(seg: np.ndarray, sr: int = SR) -> list[dict]:
    """Все значимые синусоидальные компоненты окна: частота, амплитуда, фаза."""
    win = _hann(len(seg))
    S = np.fft.rfft(seg * win)
    mag = np.abs(S) / (len(seg) / 2)
    phase = np.angle(S)
    if mag.max() <= 0:
        return []
    порог = mag.max() * ПИК_ПОРОГ_ОТН
    пики = []
    for k in range(1, len(mag) - 1):
        if mag[k] >= порог and mag[k] > mag[k - 1] and mag[k] > mag[k + 1]:
            kf, a = _параболич_пик(mag, k)
            f = kf * sr / len(seg)
            if f <= 0:
                continue
            пики.append({"частота": float(f), "амплитуда": float(a),
                         "фаза": float(phase[k])})
    return пики


def описание_окна(seg: np.ndarray, sr: int = SR) -> dict:
    """Лёгкий мгновенный набор окна (без нормировки) — для быстрых нужд."""
    win = _hann(len(seg))
    mag = np.abs(np.fft.rfft(seg * win))
    fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
    sm = mag.sum() + 1e-12
    centroid = float((fr * mag).sum() / sm)
    spread = float(np.sqrt((((fr - centroid) ** 2) * mag).sum() / sm))
    nz = mag[mag > 1e-9]
    flat = float(np.exp(np.mean(np.log(nz))) / (nz.mean() + 1e-12)) if len(nz) else 0.0
    cs = np.cumsum(mag)
    rolloff = float(fr[np.searchsorted(cs, 0.85 * cs[-1])]) if cs[-1] > 0 else 0.0
    rms = float(np.sqrt(np.mean(seg ** 2)))
    return {"centroid": centroid, "spread": spread, "flatness": min(1.0, flat),
            "rolloff85": rolloff, "rms": rms}


def мгновенные(seg: np.ndarray, sr: int = SR) -> dict:
    """ВСЕ мгновенные параметры окна — реальные величины, без нормировки.

    Считаются на одном окне (существуют в моменте). Временны́е (ритм, модуляция,
    оси, mfcc-дельты, jitter/shimmer) СЮДА НЕ входят — они уровня последовательности.
    """
    from atoms_full103 import (  # noqa: WPS433 — общий экстрактор, только чтение
        compute_spectral, compute_chroma, compute_musical, compute_perceptual,
        compute_mfcc, compute_formants, compute_pitch_hps, compute_vocal,
    )
    N = len(seg)
    sp, mag = compute_spectral(seg, sr, N)          # 18 спектральных
    flat = sp.get("spectral_flatness", 0.0)
    pitch, harm = compute_pitch_hps(mag, sr, N, flat)
    f1, f2, f3 = compute_formants(mag, sr, N, None, pitch)
    voc = compute_vocal(seg, sr, mag, pitch, harm, flat)  # VOCAL: voicing/breathiness/roughness/jitter/shimmer
    chroma = compute_chroma(mag, sr, N)             # 12 хрома
    mus = compute_musical(mag, sr, N)               # key_dominant, consonance
    perc = compute_perceptual(sp)                   # 7 перцептивных
    mfcc = compute_mfcc(mag, sr)                    # 13 mfcc
    zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
    rms = float(np.sqrt(np.mean(seg ** 2)))

    d = dict(sp)                                    # спектральные как есть
    d["pitch_hps"] = float(pitch)
    d["harmonic_ratio_hps"] = float(harm)
    d["formant_f1"], d["formant_f2"], d["formant_f3"] = float(f1), float(f2), float(f3)
    d.update({k: float(v) for k, v in voc.items()})   # VOCAL полн.: voicing/breathiness/roughness/jitter/shimmer
    for i, v in enumerate("C Cs D Ds E F Fs G Gs A As B".split()):
        d["chroma_" + v] = float(chroma[i])
    d.update({k: float(v) for k, v in mus.items()})
    d.update({k: float(v) for k, v in perc.items()})
    for i in range(len(mfcc)):
        d["mfcc_" + str(i)] = float(mfcc[i])
    d["zero_crossing_rate"] = zcr
    d["rms"] = rms
    return d


def в_кирпичи(x: np.ndarray, sr: int = SR) -> list[dict]:
    """Звук → список кирпичей-моментов. Ничего не нормируем.

    Каждый кирпич самодостаточен и обратим: хранит полный спектр окна
    (магнитуда+фаза) — этого хватает на точное восстановление ЛЮБОГО звука.

    Хвост сигнала добивается нулями до покрытия последним окном (иначе
    последние <ОКНО сэмплов выпадали из hop-сетки и corr падал с ~1.0 до ~0.99).
    """
    x = np.asarray(x, dtype=np.float64)
    if sr != SR:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)),
                      np.arange(len(x)), x)
        sr = SR
    n0 = len(x)
    # покрытие: последний старт ≥ n0 - ОКНО (с паддингом), шаг ШАГ
    if n0 < ОКНО:
        x = np.pad(x, (0, ОКНО - n0))
    else:
        хвост = (n0 - ОКНО) % ШАГ
        if хвост:
            x = np.pad(x, (0, ШАГ - хвост))
    win = _hann(ОКНО)
    кирпичи = []
    for i in range(0, len(x) - ОКНО + 1, ШАГ):
        seg = x[i:i + ОКНО]
        S = np.fft.rfft(seg * win)
        кирпичи.append({
            "t": (i + ОКНО / 2) / sr,          # центр момента
            "старт": i,
            # ЯДРО РЕКОНСТРУКЦИИ — полный спектр окна (образ→звук, любой звук):
            "спектр_магнитуда": np.abs(S),
            "спектр_фаза": np.angle(S),
            # ОБРАЗ — компактный вид (звук→образ), честен для тона:
            "пики": пики_окна(seg, sr),
            # ФЕНОТИП — ВСЕ мгновенные реальные параметры (без нормировки):
            "параметры": мгновенные(seg, sr),
            "описание": описание_окна(seg, sr),  # лёгкий срез для быстрых нужд
        })
    _добавить_форму(кирпичи, n0 / sr)   # длительность — исходная, без паддинга
    return кирпичи


def freqToHue(hz: float) -> float:
    """Цвет = частота (как в живом образе V3): 270 (низ, фиолет) → 0 (верх, красн)."""
    hz = max(20.0, min(hz, 20000.0))
    t = math.log10(hz / 20.0) / math.log10(1000.0)
    return 270.0 - min(t, 1.0) * 270.0


def _добавить_форму(атомы: list[dict], длит: float) -> None:
    """Дописать в атом форму: позиция во времени, размер, доминанта (для крестов).

    ОБРАЗ атома — это его ПИКИ, развёрнутые по частоте (Y=log частота); цвет
    каждого пика берётся из ПАЛИТРЫ ОРГАНА (живой образ V3, частота→цвет),
    а НЕ вычисляется из звука. Поэтому в атоме — только время, размер и
    доминантная частота (для группировки по крестам). Ничего не нормируем
    статистикой — только физический перевод (доля времени)."""
    if not атомы:
        return

    def доминанта(a):
        p = a["параметры"]
        f = p.get("pitch_hps") or 0.0
        if not (f and p.get("harmonic_ratio_hps", 0) > 0.4):
            f = p.get("spectral_centroid") or 20.0
        return float(max(20.0, f))

    for a in атомы:
        a["форма"] = {
            "частота_дом": round(доминанта(a), 1),         # для крестов
            "поз_x": round(a["t"] / (длит or 1.0), 4),     # ВРЕМЯ (образ по X)
            "размер": round(float(a["параметры"].get("rms", 0.0)), 4),  # амплитуда
        }


def в_клеточку(x: np.ndarray, sr: int = SR) -> dict:
    """Звук → КЛЕТОЧКА/ОРГАНИЗМ: последовательность кирпичей-моментов +
    оси, рождённые из их потока (уровень последовательности, не момента).

    Мгновенное живёт в кирпичах; временно́е (оси, тракты) — здесь, из потока.
    """
    from ядро.оси import оси_звука   # созданные ранее оси (fd, mod, самоподобие)
    x = np.asarray(x, dtype=np.float64)
    if sr != SR:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)),
                      np.arange(len(x)), x)
        sr = SR
    кирпичи = в_кирпичи(x, sr)
    # ТРАКТЫ — параметры по последовательности моментов (это и есть поток):
    def трек(ключ):
        return [float(к["параметры"].get(ключ, 0.0)) for к in кирпичи]
    тракты = {
        "t": [round(к["t"], 4) for к in кирпичи],
        "rms": трек("rms"),
        "pitch": трек("pitch_hps"),
        "centroid": трек("spectral_centroid"),
        "flatness": трек("spectral_flatness"),
    }
    # ОСИ — из потока (на сигнале = вся последовательность моментов):
    оси = оси_звука(x, sr)
    # модуляция rms-ТРАКТА (буквально из последовательности кирпичей):
    rms = np.asarray(тракты["rms"])
    кадр_гц = sr / ШАГ  # частота следования моментов
    поток_оси = {}
    if rms.size > 3 and rms.mean() > 1e-6:
        r = rms - rms.mean()
        spec = np.abs(np.fft.rfft(r * np.hanning(len(r))))
        fr = np.fft.rfftfreq(len(r), 1.0 / кадр_гц)
        поток_оси["mod_rate_трек"] = float(fr[1 + int(np.argmax(spec[1:]))]) if len(spec) > 2 else 0.0
        поток_оси["mod_depth_трек"] = float(rms.std() / (rms.mean() + 1e-9))
    return {
        "кирпичи": кирпичи,
        "оси": оси,                  # созданные ранее (на всей последовательности)
        "оси_из_трека": поток_оси,   # буквально из потока моментов
        "тракты": тракты,
        "длительность": len(x) / sr,
        "кадр_гц": кадр_гц,
    }


def в_орган(x: np.ndarray, sr: int = SR, τ_клетка: float = 0.85) -> dict:
    """Звук → ОРГАН: атомы (звук+форма) → клетки (по крестам) → орган.

    Иерархия ПРАВИЛО_СБОРКИ.md: атом → клетка → орган → организм.
    Клетка = связная компонента атомов по сильным крестам (резонанс ≥ τ_клетка,
    ядро/кресты.py, оси time/freq/harmonic). Орган = весь звук (одно явление).
    """
    from ядро.кресты import _resonance
    from ядро.оси import оси_звука
    атомы = в_кирпичи(x, sr)
    N = len(атомы)
    U = [{
        "birth": a["t"], "freq": a["форма"]["частота_дом"],
        "harmonic_index": 1 if a["параметры"].get("harmonic_ratio_hps", 0) > 0.5 else 0,
        "harmonicity": a["параметры"].get("harmonic_ratio_hps", 0),
    } for a in атомы]

    def связь(i, j):
        return max(_resonance(U[i], U[j], ax) for ax in ("time", "freq", "harmonic"))

    par = list(range(N))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for a in атомы:
        a["форма"]["кресты"] = []
    for i in range(N):
        for j in range(i + 1, N):
            if связь(i, j) >= τ_клетка:
                par[find(i)] = find(j)
                атомы[i]["форма"]["кресты"].append(j)
                атомы[j]["форма"]["кресты"].append(i)
    groups: dict[int, list[int]] = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    клетки = sorted(groups.values(), key=lambda g: min(g))

    xa = np.asarray(x, dtype=np.float64)
    оси = оси_звука(xa, sr)
    return {
        "атомы": атомы,          # каждый: звук (спектр+58 параметров+пики) + форма (+кресты)
        "клетки": клетки,        # списки индексов атомов (связные по крестам)
        "орган": {
            "атомов": N, "клеток": len(клетки), "оси": оси,
            "длительность": len(xa) / sr, "τ_клетка": τ_клетка,
        },
    }


def из_кирпичей(кирпичи: list[dict], длина: int | None = None, sr: int = SR) -> np.ndarray:
    """Кирпичи → звук. Точное восстановление из полного спектра окна,
    overlap-add с делением на Σw. Обратимость образ→звук для любого звука."""
    if not кирпичи:
        return np.zeros(1)
    n = длина if длина else (кирпичи[-1]["старт"] + ОКНО)
    y = np.zeros(n + ОКНО)
    wsum = np.zeros(n + ОКНО)
    win = _hann(ОКНО)
    for к in кирпичи:
        S = к["спектр_магнитуда"] * np.exp(1j * к["спектр_фаза"])
        rec = np.fft.irfft(S, ОКНО)   # обратно оконный сегмент (x·w)
        i = к["старт"]
        y[i:i + ОКНО] += rec          # rec ≈ x·w_анализа
        wsum[i:i + ОКНО] += win       # делим на Σw_анализа → x
    wsum[wsum < 1e-6] = 1.0
    return (y / wsum)[:n]

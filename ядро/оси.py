# -*- coding: utf-8 -*-
"""
оси.py — фрактальные и временны́е оси звука для Тринити V3.

Отдельный модуль, НЕ трогает фаза.py. Подключается к ядру:
    from оси import оси_звука
    p = оси_звука(x, sr)   # x — numpy-массив моно, sr — частота дискретизации

Возвращает «паспорт осей» (часть единого паспорта звука):
    {fd, selfsim_r2, nestedness, mod_depth, mod_rate}

Все функции воспроизводят значения, проверенные в сессии 15–16 июня 2026:
  гласная «О протяжное» ~ fd 0.09 · шум «С протяжное» ~ fd 1.06
  round-trip FD r=0.974 · второй диктор r=0.90 (полюса инвариантны).

Зависимости: numpy, scipy.
"""
import numpy as np
from scipy import signal


# ── подготовка ────────────────────────────────────────────────────────────
def обрезать_тишину(x, доля=0.15):
    """Обрезает тихие края по огибающей Гильберта (как в анализе буквицы)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 32:
        return x
    env = np.abs(signal.hilbert(x))
    thr = доля * env.max()
    on = np.where(env > thr)[0]
    if len(on) > 10:
        x = x[on[0]:on[-1] + 1]
    return x


# ── 1. фрактальная размерность Хигучи + самоподобие ────────────────────────
def higuchi(x, kmax=12):
    """
    Возвращает (FD, R²):
      FD  — фрактальная размерность Хигучи (шероховатость/многомасштабность),
            непрерывный спектр буквицы 0.05 (гласные) → 1.06 (шум).
      R²  — линейность лог-лог графика = насколько это ИСТИННЫЙ фрактал
            (самоподобие); 0.77 у периодичных, ~1.0 у шума/стихий.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    P = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            Lk.append(np.sum(np.abs(np.diff(x[idx]))) * (N - 1) / ((len(idx) - 1) * k))
        if Lk:
            P.append((np.log(1.0 / k), np.log(np.mean(Lk))))
    P = np.array(P)
    a, b = np.polyfit(P[:, 0], P[:, 1], 1)
    fit = a * P[:, 0] + b
    ss = 1.0 - np.sum((P[:, 1] - fit) ** 2) / np.sum((P[:, 1] - P[:, 1].mean()) ** 2)
    return float(a), float(ss)


# ── 2. взаимовложенность (сцепка масштабов во времени) ─────────────────────
def вложенность(x, sr):
    """
    Связаны ли масштабы: средняя корреляция огибающих октавных полос.
    Гласные ~0.9 (гармоники в формантах), ровный шум ~0 (масштабы независимы).
    ВАЖНО: это ВРЕМЕННА́Я сцепка, не пространственная. Антикоррелирует с FD (r=-0.57).
    """
    x = np.asarray(x, dtype=float)
    полосы = [(120, 240), (240, 480), (480, 960), (960, 1920), (1920, 3840), (3840, 7500)]
    envs = []
    for lo, hi in полосы:
        hi = min(hi, sr / 2 - 1)
        if hi <= lo:
            continue
        sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        bp = signal.sosfiltfilt(sos, x)
        if np.sqrt(np.mean(bp ** 2)) < 0.005:
            continue
        e = np.abs(signal.hilbert(bp))
        if len(e) > 205:
            e = signal.savgol_filter(e, 201, 2)
        envs.append((e - e.mean()) / (e.std() + 1e-9))
    if len(envs) < 2:
        return 0.0
    cs = []
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            n = min(len(envs[i]), len(envs[j]))
            cs.append(np.corrcoef(envs[i][:n], envs[j][:n])[0, 1])
    return float(np.mean(cs))


# ── 3. временна́я ось: модуляция (жизнь огибающей) ──────────────────────────
def модуляция(x, sr):
    """
    Возвращает (глубина, ритм):
      глубина — насколько «дышит» громкость (std/mean огибающей); 0.25..2.0.
      ритм    — ведущая частота пульсации (Гц), пик модуляционного спектра.
    Это та ось «жизни/формирования», которую теряет машинный синтез (см. остаток).
    """
    x = np.asarray(x, dtype=float)
    e = np.abs(signal.hilbert(x))
    if len(e) > 205:
        e = signal.savgol_filter(e, 201, 2)
    depth = float(e.std() / (e.mean() + 1e-9))
    ec = e - e.mean()
    fm, Pe = signal.welch(ec, sr, nperseg=min(8192, len(ec)))
    m = (fm >= 0.5) & (fm <= 40)
    rate = float(fm[m][np.argmax(Pe[m])]) if m.any() and Pe[m].max() > 0 else 0.0
    return depth, rate


# ── единый паспорт осей ────────────────────────────────────────────────────
def оси_звука(x, sr, обрезать=True):
    """
    Главная функция. Возвращает паспорт новых осей для единого паспорта звука.
    Подключать в ядро рядом с фазой: один анализатор → один паспорт → одно тело.
    """
    x = np.asarray(x, dtype=float)
    мх = np.max(np.abs(x))
    if мх > 0:
        x = x / мх
    if обрезать:
        x = обрезать_тишину(x)
    fd, r2 = higuchi(x)
    nst = вложенность(x, sr)
    depth, rate = модуляция(x, sr)
    return {
        "fd": round(fd, 3),
        "selfsim_r2": round(r2, 3),
        "nestedness": round(nst, 3),
        "mod_depth": round(depth, 3),
        "mod_rate": round(rate, 1),
    }


# ── зоны FD (устойчивы к голосу: см. второй диктор) ─────────────────────────
def зона_fd(fd):
    """Крупная разбивка FD, инвариантная к диктору (полюса ~97%).
    Мелкие 5 семейств между дикторами не держатся — используем 2-3 зоны."""
    if fd < 0.30:
        return "тон-полюс"
    if fd < 0.60:
        return "середина"
    return "фрактал-полюс"


# ── самопроверка ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, subprocess, tempfile, os
    from scipy.io import wavfile

    def загрузить(path, sr=16000):
        import shutil
        f = tempfile.mktemp(suffix=".wav")
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(sr), f],
                           capture_output=True)
        else:  # macOS: встроенный декодер без ffmpeg
            subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{sr}", "-c", "1", path, f],
                           capture_output=True)
        s, x = wavfile.read(f)
        os.remove(f)
        if x.ndim > 1:
            x = x.mean(1)
        return x.astype(float), s

    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            x, sr = загрузить(p)
            print(os.path.basename(p), "→", оси_звука(x, sr))
    else:
        print("Использование: python3 оси.py файл1.m4a файл2.wav ...")
        print("Функции: higuchi(x), вложенность(x,sr), модуляция(x,sr), оси_звука(x,sr)")

# -*- coding: utf-8 -*-
"""
экзамен_осей.py — round-trip проверка фрактальных осей для Тринити V3.

Проверяет, что новые оси (FD и др.) — реальное свойство образа, а не артефакт:
  звук → координата (ось) → ресинтез из спектра → пере-измерение оси → сверка.

Эталон, полученный в сессии 15–16 июня 2026 (вся буквица, 37 букв):
  FD round-trip: Pearson r = 0.974, MAE = 0.043, семейство 30/37.
  Если прогон даёт близкие числа — анализатор и контракт целы.

Подключает оси.py из ядра. Зависимости: numpy, scipy (+ ffmpeg для загрузки).

Использование:
  python3 экзамен_осей.py <папка с аудио>      # агрегатный round-trip по папке
  python3 экзамен_осей.py файл1 файл2 ...       # по отдельным файлам
"""
import os, sys
import numpy as np
from scipy import signal

# импорт оси.py из соседней папки ядро/
_ЯДРО = os.path.join(os.path.dirname(__file__), "..", "ядро")
sys.path.insert(0, os.path.abspath(_ЯДРО))
from оси import оси_звука, higuchi, обрезать_тишину   # noqa: E402


def _гармоничность(x, sr):
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac0 = ac[0] + 1e-12
    lo, hi = int(sr / 400), int(sr / 70)
    seg = ac[lo:hi] / ac0
    return float(seg.max()) if len(seg) else 0.0


def ресинтез(x, sr):
    """Машинный двойник: магнитудный спектр сохранён, фаза когерентна (тон)
    или случайна (шум) — как в контракте обратимости round_trip_score."""
    X = np.fft.rfft(x)
    mag = np.abs(X)
    harm = _гармоничность(x, sr)
    if harm > 0.45:
        phase = np.zeros_like(mag)                     # тон → периодика
    else:
        phase = np.random.uniform(-np.pi, np.pi, len(mag))  # шум → случайно
    y = np.fft.irfft(mag * np.exp(1j * phase), n=len(x))
    m = np.max(np.abs(y))
    return y / m if m > 0 else y


def round_trip_осей(x, sr):
    """Для одного звука: оси оригинала, оси ресинтеза, отклонение FD."""
    x = np.asarray(x, dtype=float)
    mx = np.max(np.abs(x))
    if mx > 0:
        x = x / mx
    x = обрезать_тишину(x)
    о = оси_звука(x, sr, обрезать=False)
    y = ресинтез(x, sr)
    fd2, r22 = higuchi(y)
    return {
        "fd": о["fd"], "fd_ресинтез": round(fd2, 3),
        "fd_дельта": round(abs(о["fd"] - fd2), 3),
        "selfsim_r2": о["selfsim_r2"], "nestedness": о["nestedness"],
    }


def _загрузить(path, sr=16000):
    import subprocess, tempfile, shutil
    from scipy.io import wavfile
    f = tempfile.mktemp(suffix=".wav")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(sr), f],
                       capture_output=True)
    else:  # macOS: встроенный декодер без ffmpeg
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{sr}", "-c", "1", path, f],
                       capture_output=True)
    s, x = wavfile.read(f)
    os.remove(f)
    x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(1)
    return x, s


def экзамен_папки(folder):
    """Агрегатный round-trip FD по всем аудио в папке: r, MAE, балл."""
    import glob
    np.random.seed(0)
    pairs = []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if not p.lower().endswith((".m4a", ".wav", ".mp3", ".ogg")):
            continue
        x, sr = _загрузить(p)
        r = round_trip_осей(x, sr)
        pairs.append((os.path.basename(p), r["fd"], r["fd_ресинтез"]))
    if len(pairs) < 3:
        return {"ошибка": "мало файлов", "n": len(pairs)}
    a = np.array([p[1] for p in pairs]); b = np.array([p[2] for p in pairs])
    r = float(np.corrcoef(a, b)[0, 1])
    mae = float(np.mean(np.abs(a - b)))
    балл = round(max(0.0, r), 3)
    return {"n": len(pairs), "pearson_r": round(r, 3), "MAE": round(mae, 3),
            "балл": балл, "эталон_r": 0.974}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python3 экзамен_осей.py <папка>     # агрегатный round-trip")
        print("  python3 экзамен_осей.py файл ...     # по файлам")
        sys.exit(0)
    arg = sys.argv[1]
    if os.path.isdir(arg):
        res = экзамен_папки(arg)
        print("ЭКЗАМЕН ОСЕЙ (round-trip FD по папке):")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        for p in sys.argv[1:]:
            x, sr = _загрузить(p)
            print(os.path.basename(p), "→", round_trip_осей(x, sr))

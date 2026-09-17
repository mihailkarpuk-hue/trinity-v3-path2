# -*- coding: utf-8 -*-
"""
прогон_живой_буквицы.py — поток по 37 живым буквам:
  звук → оси (fd, selfsim_r2, nestedness, mod_depth, mod_rate) → round-trip fd → статистика.
Зависимости: numpy, scipy, ffmpeg.
"""
import os, sys, glob, json
import numpy as np

ЗДЕСЬ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ЗДЕСЬ, "ядро"))
sys.path.insert(0, os.path.join(ЗДЕСЬ, "экзамен"))

from оси import оси_звука, higuchi, обрезать_тишину          # noqa
from экзамен_осей import _загрузить, round_trip_осей           # noqa

ПАПКА = os.path.join(ЗДЕСЬ, "данные", "клеточки", "буквица_живая")

def имя(p):
    return os.path.splitext(os.path.basename(p))[0].replace("живая_", "")

def main():
    файлы = sorted(glob.glob(os.path.join(ПАПКА, "*.m4a")))
    строки, fd_in, fd_out = [], [], []
    for p in файлы:
        try:
            x, sr = _загрузить(p)
            o = оси_звука(x, sr)
            rt = round_trip_осей(x, sr)   # {'fd_in','fd_out',...}
            строки.append({
                "буква": имя(p),
                "fd": round(o["fd"], 3),
                "selfsim_r2": round(o["selfsim_r2"], 3),
                "nestedness": round(o["nestedness"], 3),
                "mod_depth": round(o["mod_depth"], 3),
                "mod_rate": round(o["mod_rate"], 2),
                "fd_rt": round(rt.get("fd_ресинтез", float("nan")), 3),
            })
            fd_in.append(o["fd"])
            fd_out.append(строки[-1]["fd_rt"])
        except Exception as e:
            строки.append({"буква": имя(p), "ошибка": str(e)[:80]})
    ок = [s for s in строки if "ошибка" not in s]
    fd_in = np.array([s["fd"] for s in ок], float)
    fd_out = np.array([s["fd_rt"] for s in ок], float)
    маска = ~np.isnan(fd_out)
    r = float(np.corrcoef(fd_in[маска], fd_out[маска])[0, 1]) if маска.sum() > 1 else float("nan")
    mae = float(np.mean(np.abs(fd_in[маска] - fd_out[маска]))) if маска.sum() else float("nan")

    def сорт(ключ):
        return sorted(ок, key=lambda s: s[ключ])

    сводка = {
        "n_букв": len(ок),
        "round_trip_fd": {"pearson_r": round(r, 3), "mae": round(mae, 3)},
        "fd": {"мин": round(float(fd_in.min()), 3), "макс": round(float(fd_in.max()), 3), "сред": round(float(fd_in.mean()), 3)},
        "nestedness": {"мин": round(min(s["nestedness"] for s in ок), 3), "макс": round(max(s["nestedness"] for s in ок), 3), "сред": round(float(np.mean([s["nestedness"] for s in ок])), 3)},
        "fd_vs_nest_r": round(float(np.corrcoef([s["fd"] for s in ок], [s["nestedness"] for s in ок])[0, 1]), 3),
        "топ_фрактальные": [(s["буква"], s["fd"]) for s in сорт("fd")[-6:][::-1]],
        "топ_вложенные": [(s["буква"], s["nestedness"]) for s in сорт("nestedness")[-6:][::-1]],
    }
    out = {"сводка": сводка, "строки": строки}
    with open(os.path.join(ЗДЕСЬ, "статистика_живой_буквицы.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(сводка, ensure_ascii=False, indent=2))
    print("\nЗаписано: статистика_живой_буквицы.json  (строк:", len(строки), ")")

if __name__ == "__main__":
    main()

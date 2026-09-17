# -*- coding: utf-8 -*-
"""построить_паспорта.py — встроить полный 104-паспорт в каждую клетку каталога.
Прогоняет scripts/atoms_full103.py (103 параметра) + ядро/оси.py (5 фракт./врем. осей)
по звуку каждой клетки и дописывает поле `параметры104`. Атом не трогает."""
import os, sys, json, subprocess, tempfile, shutil
import numpy as np
from scipy.io import wavfile

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
ПРОЕКТ = os.path.dirname(КОРЕНЬ)
sys.path.insert(0, os.path.join(ПРОЕКТ, "scripts"))
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))
from atoms_full103 import analyze_full_103, list_103_keys
from оси import оси_звука

КЛЕТКИ = os.path.join(КОРЕНЬ, "данные", "клеточки")

def загрузить(path, sr=16000):
    f = tempfile.mktemp(suffix=".wav")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg","-y","-i",path,"-ac","1","-ar",str(sr),f],capture_output=True)
    else:
        subprocess.run(["afconvert","-f","WAVE","-d",f"LEI16@{sr}","-c","1",path,f],capture_output=True)
    s,x = wavfile.read(f); os.remove(f); x=x.astype(float)
    if x.ndim>1: x=x.mean(1)
    m=np.max(np.abs(x));  x=x/m if m>0 else x
    return x, s

кат = os.path.join(КЛЕТКИ, "каталог.json")
d = json.load(open(кат, encoding="utf-8"))
items = d if isinstance(d, list) else next((v for v in d.values() if isinstance(v, list)), [])
print(f"клеток: {len(items)} · ключей 103: {len(list_103_keys())}")

ок, нет = 0, 0
for i, it in enumerate(items):
    z = it.get("звук", "")
    p = os.path.join(КЛЕТКИ, z) if z else ""
    if not p or not os.path.exists(p):
        нет += 1; continue
    try:
        x, sr = загрузить(p)
        п103 = analyze_full_103(x, sr)
        оси = оси_звука(x, sr)
        it["параметры104"] = {**п103, **оси}   # 103 + 5 = 108 полей
        ок += 1
    except Exception as e:
        нет += 1
        if нет <= 3: print(" сбой:", z, "→", repr(e)[:80])
    if (i+1) % 25 == 0: print(f"  …{i+1}/{len(items)}")

json.dump(d, open(кат, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"ГОТОВО: встроено в {ок} клеток, пропущено {нет}")
if items and "параметры104" in items[0]:
    ex = items[0]["параметры104"]
    print(f"пример [{items[0].get('id')}]: полей={len(ex)}; "
          f"жизнь: jitter={ex.get('jitter')}, shimmer={ex.get('shimmer')}, "
          f"nestedness={ex.get('nestedness')}, mod_depth={ex.get('mod_depth')}")

# ── ШОВ 6: экзамен осей как гейт ──
ПОРОГ_R = 0.90
буквица = os.path.join(КЛЕТКИ, "буквица_живая")
if os.path.isdir(буквица):
    import subprocess
    экз = os.path.join(КОРЕНЬ, "экзамен", "экзамен_осей.py")
    print("\n[шов 6] экзамен round-trip осей (буквица_живая)…")
    r = subprocess.run([sys.executable, экз, буквица], capture_output=True, text=True)
    print(r.stdout or r.stderr)
    if r.returncode != 0:
        print(f"⚠️  ЭКЗАМЕН НЕ ПРОЙДЕН (код {r.returncode}). Каталог записан, проверь оси.")
    else:
        print("✅ экзамен осей завершён")
else:
    print("[шов 6] буквица_живая не найдена — экзамен пропущен")

# -*- coding: utf-8 -*-
"""Шаг 1: собрать ОДНУ клетку в полном виде — проба структуры (клетка 104 + атом 28).
Не трогает остальные клетки. Пишет образец в данные/клеточки/_образец_А_полная.json."""
import os, sys, json, subprocess, tempfile, shutil
import numpy as np
from scipy.io import wavfile

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__)); ПРОЕКТ = os.path.dirname(КОРЕНЬ)
sys.path.insert(0, os.path.join(ПРОЕКТ, "scripts")); sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))
import types as _t; sys.modules["soundfile"]=_t.ModuleType("soundfile")
from atoms_core28 import extract_atoms_28
from atoms_full103 import analyze_full_103
from оси import оси_звука

ЗВУК = os.path.join(КОРЕНЬ, "данные", "алфавит ", "А.m4a")

def wav(path, sr):
    f = tempfile.mktemp(suffix=".wav")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg","-y","-i",path,"-ac","1","-ar",str(sr),f],capture_output=True)
    else:
        subprocess.run(["afconvert","-f","WAVE","-d",f"LEI16@{sr}","-c","1",path,f],capture_output=True)
    s,x = wavfile.read(f); os.remove(f); x=x.astype(float)
    if x.ndim>1: x=x.mean(1)
    return x, s

# атомы — на 44.1 кГц (как задумано в core28)
x44, sr44 = wav(ЗВУК, 44100); m=np.max(np.abs(x44)); x44n = x44/m if m else x44
atoms, parent_legacy = extract_atoms_28(x44n, sr44, "Тринити_V3/данные/алфавит /А.m4a")
# клеточный паспорт 104 — на 16 кГц (как наши параметры104)
x16, sr16 = wav(ЗВУК, 16000); m=np.max(np.abs(x16)); x16n = x16/m if m else x16
п103 = analyze_full_103(x16n, sr16); оси = оси_звука(x16n, sr16)

клетка = {
    "version": "2.0-full",
    "core_size": 28,
    "parent_path": "Тринити_V3/данные/алфавит /А.m4a",
    "parent_duration_sec": round(len(x44)/sr44, 4),
    "parent_params_full": {**п103, **оси},     # ПОЛНЫЕ 104 (103 + 5 наших осей)
    "parent_params_legacy11": parent_legacy,    # 11 — для совместимости/диагностики
    "atoms_count": len(atoms),
    "атомы": atoms,
}
out = os.path.join(КОРЕНЬ, "данные", "клеточки", "_образец_А_полная.json")
json.dump(клетка, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=1)

# ── ПРОВЕРКА СТРУКТУРЫ ──
a0 = atoms[0]
print("=== КЛЕТКА «А» В ПОЛНОМ ВИДЕ ===")
print(f"атомов: {len(atoms)}")
print(f"полей в атоме: {len(a0)}  (цель 28)")
print(f"  ключи атома: {sorted(a0.keys())}")
print(f"клеточный паспорт parent_params_full: {len(клетка['parent_params_full'])} полей (цель 108=103+5)")
print(f"  оси клетки: fd={оси['fd']} r2={оси['selfsim_r2']} nestedness={оси['nestedness']} mod_depth={оси['mod_depth']}")
print(f"размер файла: {os.path.getsize(out)//1024} КБ")
print(f"сохранено: данные/клеточки/_образец_А_полная.json")

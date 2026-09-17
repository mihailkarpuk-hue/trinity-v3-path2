# -*- coding: utf-8 -*-
"""Шаг 3: все 113 клеток в полном виде (атом 28 + клетка 104) → клеточки_полные/.
Не трогает рабочие клетки. Собирает статистику."""
import os, sys, json, subprocess, tempfile, shutil, types, time
import numpy as np
from scipy.io import wavfile
sys.modules["soundfile"] = types.ModuleType("soundfile")
КОРЕНЬ=os.path.dirname(os.path.abspath(__file__)); ПРОЕКТ=os.path.dirname(КОРЕНЬ)
sys.path.insert(0,os.path.join(ПРОЕКТ,"scripts")); sys.path.insert(0,os.path.join(КОРЕНЬ,"ядро"))
from atoms_core28 import extract_atoms_28
from atoms_full103 import analyze_full_103
from оси import оси_звука
КЛ=os.path.join(КОРЕНЬ,"данные","клеточки"); ВЫХ=os.path.join(КЛ,"клеточки_полные"); os.makedirs(ВЫХ,exist_ok=True)

def wav(path,sr):
    f=tempfile.mktemp(suffix=".wav")
    if shutil.which("ffmpeg"): subprocess.run(["ffmpeg","-y","-i",path,"-ac","1","-ar",str(sr),f],capture_output=True)
    else: subprocess.run(["afconvert","-f","WAVE","-d",f"LEI16@{sr}","-c","1",path,f],capture_output=True)
    s,x=wavfile.read(f); os.remove(f); x=x.astype(float)
    if x.ndim>1: x=x.mean(1)
    return x,s

d=json.load(open(os.path.join(КЛ,"каталог.json"),encoding="utf-8"))
items=d if isinstance(d,list) else next((v for v in d.values() if isinstance(v,list)),[])
t0=time.time(); ok=0; fail=0; stats=[]
for i,it in enumerate(items):
    z=it.get("звук",""); p=os.path.join(КЛ,z) if z else ""
    if not p or not os.path.exists(p): fail+=1; continue
    try:
        x44,sr44=wav(p,44100); m=np.max(np.abs(x44)); x44=x44/m if m else x44
        atoms,_=extract_atoms_28(x44,sr44,z)
        x16,sr16=wav(p,16000); m=np.max(np.abs(x16)); x16=x16/m if m else x16
        п103=analyze_full_103(x16,sr16); оси=оси_звука(x16,sr16)
        клетка={"version":"2.0-full","core_size":28,"id":it.get("id"),"parent_path":z,
                "parent_params_full":{**п103,**оси},"atoms_count":len(atoms),"атомы":atoms}
        json.dump(клетка,open(os.path.join(ВЫХ,(it.get("id") or f"cell{i}")+".json"),"w",encoding="utf-8"),ensure_ascii=False)
        ph="тон" if п103.get("harmonic_ratio",0)>0.45 else ("шум" if п103.get("harmonic_ratio",0)<0.2 else "переход")
        stats.append(dict(id=it.get("id"),группа=it.get("группа"),atoms=len(atoms),
            fd=оси["fd"],nest=оси["nestedness"],mod=оси["mod_depth"],
            jitter=round(float(п103.get("jitter",0)),3),shimmer=round(float(п103.get("shimmer",0)),3),
            centroid=round(float(п103.get("spectral_centroid",0))),harm=round(float(п103.get("harmonic_ratio",0)),2),phase=ph,
            поля_атома=len(atoms[0]) if atoms else 0,поля_паспорта=len({**п103,**оси})))
        ok+=1
    except Exception as e:
        fail+=1
        if fail<=3: print("сбой:",z,"→",repr(e)[:70])
    if (i+1)%25==0: print(f"  …{i+1}/{len(items)}  ({time.time()-t0:.0f}с)")

json.dump(stats,open(os.path.join(КОРЕНЬ,"данные","статистика_полные.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(f"\n=== ГОТОВО: {ok} клеток, сбоев {fail}, время {time.time()-t0:.0f}с ===")
print(f"папка: данные/клеточки/клеточки_полные/  ({sum(os.path.getsize(os.path.join(ВЫХ,f)) for f in os.listdir(ВЫХ))//1024//1024} МБ)")

# -*- coding: utf-8 -*-
"""Добавить ЖИВУЮ буквицу (37 записей голосом) как клетки V3 — увидеть образ
живых звуков алфавита. Через единый паспорт (фаза.паспорт, шов 1) + атомы 28."""
import os, sys, json, glob, shutil, subprocess, tempfile, types
import numpy as np
from scipy.io import wavfile
sys.modules["soundfile"]=types.ModuleType("soundfile")
КОРЕНЬ=os.path.dirname(os.path.abspath(__file__)); ПРОЕКТ=os.path.dirname(КОРЕНЬ)
sys.path.insert(0,os.path.join(ПРОЕКТ,"scripts")); sys.path.insert(0,os.path.join(КОРЕНЬ,"ядро"))
from atoms_core28 import extract_atoms_28
from фаза import паспорт   # ← ШОВ 1: единый анализатор

СРС=os.path.join(КОРЕНЬ,"данные","алфавит ")
КЛ=os.path.join(КОРЕНЬ,"данные","клеточки")
ВЫХ=os.path.join(КЛ,"буквица_живая"); os.makedirs(ВЫХ,exist_ok=True)

def wav(path,sr):
    f=tempfile.mktemp(suffix=".wav")
    if shutil.which("ffmpeg"): subprocess.run(["ffmpeg","-y","-i",path,"-ac","1","-ar",str(sr),f],capture_output=True)
    else: subprocess.run(["afconvert","-f","WAVE","-d",f"LEI16@{sr}","-c","1",path,f],capture_output=True)
    s,x=wavfile.read(f); os.remove(f); x=x.astype(float)
    if x.ndim>1: x=x.mean(1)
    return x,s

d=json.load(open(os.path.join(КЛ,"каталог.json"),encoding="utf-8"))
ключ=next(k for k,v in d.items() if isinstance(v,list)); lst=d[ключ]
# убрать прежние живые (если перезапуск)
lst[:]= [c for c in lst if c.get("группа")!="буквица_живая"]

добавлено=0
for p in sorted(glob.glob(СРС+"/*.m4a")):
    имя=os.path.basename(p)[:-4]; sid="живая_"+имя.replace(" ","_")
    x44,sr44=wav(p,44100); m=np.max(np.abs(x44)); x44=x44/m if m else x44
    atoms,_=extract_atoms_28(x44,sr44,"буквица_живая/"+имя)
    x16,sr16=wav(p,16000); m=np.max(np.abs(x16)); x16=x16/m if m else x16
    пасп=паспорт(x16,sr16)                          # единый паспорт
    p104=пасп.get("параметры104",{})
    дл=round(len(x44)/sr44,3)
    # копия звука для проигрывания + файл атомов
    shutil.copy2(p, os.path.join(ВЫХ, sid+".m4a"))
    json.dump({"version":"2.0-full","длительность_сек":дл,"atoms_count":len(atoms),
               "atoms":atoms,"parent_params_full":p104},
              open(os.path.join(ВЫХ,sid+".json"),"w",encoding="utf-8"),ensure_ascii=False)
    lst.append({
        "id":sid,"название":имя+" (живой голос)","группа":"буквица_живая",
        "длительность_сек":дл,"число_атомов":len(atoms),"число_связей":0,
        "основной_элемент":"голос","звук":"буквица_живая/"+sid+".m4a",
        "сцена":"", "атомы":"буквица_живая/"+sid+".json","решётка":"буквица_живая/"+sid+".json",
        "параметры104":p104,
    })
    добавлено+=1

json.dump(d,open(os.path.join(КЛ,"каталог.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(f"добавлено живых букв: {добавлено}")
print(f"группа: буквица_живая · звук+атомы в данные/клеточки/буквица_живая/")
# проверка одной
ex=[c for c in lst if c["группа"]=="буквица_живая"][0]
print(f"пример: {ex['название']} · атомов {ex['число_атомов']} · fd={ex['параметры104'].get('fd')} вложенность={ex['параметры104'].get('nestedness')}")

# -*- coding: utf-8 -*-
"""ПЕРЕСБОРКА ЖИВЫХ БУКВ — заменяет синтетические буквы на записанный голос.
Кладёшь записи в Тринити_V3/буквы_живые/ (a.wav, b.mp3, s.m4a …), запускаешь:
    python3 scripts/пересборка_живых_букв.py
Для каждой найденной буквы: декод → реальный звук → атомизация → фаза (канон) →
образ по архетипу буквицы (форма+цвет) с настоящим тембром → запись клетки.
Самодостаточно (numpy + ffmpeg)."""
import os, glob, json, wave, math, subprocess, random
import numpy as np
SR=22050; N=2048; HOP=int(SR*0.016)
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Тринити_V3
ВХОД=os.path.join(ROOT,"буквы_живые"); ET=os.path.join(ROOT,"данные","клеточки","эталоны")
КАРТА=os.path.join(ROOT,"данные","буквица_карта.json")
TR={'А':'a','Б':'b','В':'v','Г':'g','Д':'d','Е':'e','Ё':'yo','Ж':'zh','З':'z','И':'i','Й':'j','К':'k','Л':'l','М':'m','Н':'n','О':'o','П':'p','Р':'r','С':'s','Т':'t','У':'u','Ф':'f','Х':'h','Ц':'c','Ч':'ch','Ш':'sh','Щ':'shch','Ъ':'hard','Ы':'y','Ь':'soft','Э':'eh','Ю':'yu','Я':'ya'}

def load(p):
    with wave.open(p) as w:
        sr=w.getframerate(); ch=w.getnchannels(); x=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(float)/32768
    return (x.reshape(-1,2).mean(1) if ch==2 else x), sr
def decode(src,slug,dur=1.2):
    real=os.path.join(ET,f"bukva_{slug}_real.wav")
    subprocess.run(["ffmpeg","-y","-v","quiet","-i",src,"-t",str(dur),"-ar","44100","-ac","2","-af","loudnorm=I=-16:TP=-1.5",real],check=True)
    ana=f"/tmp/_живбукв_{slug}.wav"
    subprocess.run(["ffmpeg","-y","-v","quiet","-i",src,"-t",str(dur),"-ar","22050","-ac","1",ana],check=True)
    x,_=load(ana); x=x/(np.abs(x).max()+1e-9)*0.9
    tmp=f"/tmp/_живбукв_n_{slug}.wav"
    with wave.open(tmp,'w') as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((x*32767).astype(np.int16).tobytes())
    x,_=load(tmp); return x, real
def атомизировать(x):
    import sys
    _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _r not in sys.path:
        sys.path.insert(0, _r)
    from ядро.атомизация import атомизировать as _ат
    return [(a["birth"], a["freq"], a["amp"], a["harmonicity"]) for a in _ат(x, SR)]
def hps_harm(mag):
    bh=SR/N; mn=max(2,int(60/bh)); mx=min(len(mag)-1,int(3000/bh)); hps=mag[mn:mx+1].copy()
    for h in (2,3,4):
        end=min(len(mag),(mx+1)*h); dec=mag[mn*h:end:h]; n=min(len(hps),len(dec)); hps[:n]*=dec[:n]
    best=int(np.argmax(hps))+mn; pitch=best*bh
    if pitch<60: return 0.0
    et=float(mag.sum()+1e-9); eh=sum(float(mag[int(round(pitch*k/bh))]) for k in range(1,8) if int(round(pitch*k/bh))<len(mag))
    return min(1.0,eh/et*6)
def фаза_кадры(x):
    win=0.5*(1-np.cos(2*np.pi*np.arange(N)/(N-1))); T=[];H=[]
    for i in range(0,max(1,len(x)-N),HOP):
        mag=np.abs(np.fft.rfft(x[i:i+N]*win)); nz=mag[mag>1e-9]
        flat=float(np.exp(np.mean(np.log(nz)))/(nz.mean()+1e-12)) if len(nz) else 1.0
        T.append(i/SR); H.append(0.0 if flat>0.3 else hps_harm(mag))
    return np.array(T),np.array(H)
def ph(h): return 'тон' if h>0.6 else ('шум' if h<0.05 else 'переход')
def params104(x):
    win=0.5*(1-np.cos(2*np.pi*np.arange(N)/(N-1))); fr=np.fft.rfftfreq(N,1/SR)
    S=np.array([np.abs(np.fft.rfft(x[i:i+N]*win))/(N/2) for i in range(0,max(1,len(x)-N),HOP)]); avg=S.mean(0)+1e-12
    cen=float((fr*avg).sum()/avg.sum()); flat=float(np.exp(np.log(avg).mean())/avg.mean())
    low=float(avg[fr<500].sum()/avg.sum()); high=float(avg[fr>2000].sum()/avg.sum())
    pk=[(avg[i],fr[i]) for i in range(2,len(avg)-1) if avg[i]>avg[i-1] and avg[i]>avg[i+1] and fr[i]>150]; pk.sort(reverse=True)
    f1=int(pk[0][1]) if pk else 0; f2=int(pk[1][1]) if len(pk)>1 else 0
    return {"centroid":round(cen),"rms":round(float(np.sqrt((x**2).mean())),4),"harmonicity":round(float(max(0,1-flat)),3),
            "flatness":round(flat,3),"low":round(low,3),"high":round(high,3),"formantF1":f1,"formantF2":f2,"источник":"живой голос"}
def G(s=1.0): return random.gauss(0,s)
def форма(tag,n):
    p=[]
    for _ in range(n):
        x=y=z=0.0
        if tag in ('восхождение','огонь'): y=random.uniform(-1.05,1.15); k=1-(y+1)/2.2*0.6; x=G(0.16)*k; z=G(0.16)*k
        elif tag in ('дыхание','кольцо/объём','покой/симметрия'):
            u=random.uniform(0,6.283); v=math.acos(random.uniform(-1,1)); r=random.random()**(1/3)*0.9
            x=math.sin(v)*math.cos(u)*r; y=math.cos(v)*r; z=math.sin(v)*math.sin(u)*r
        elif tag=='пульс': x=G(0.4); y=G(0.4); z=G(0.32)
        elif tag=='волны': x=random.uniform(-1,1); z=random.uniform(-0.6,0.6); y=0.45*math.sin(x*3)+G(0.05)
        elif tag=='удар':
            if random.random()<0.7: x=G(0.05); y=random.uniform(-1.1,1.1); z=G(0.05)
            else: a=random.uniform(0,6.283); r=random.random()*0.9; x=math.cos(a)*r; y=math.sin(a)*r*0.5; z=G(0.1)
        elif tag=='ветвление':
            if random.random()<0.4: x=G(0.06); y=random.uniform(-1.1,0.1); z=G(0.06)
            else: s=random.choice([-1,1]); tt=random.random(); x=s*tt*0.9; y=-0.1+tt+G(0.05); z=G(0.12)
        elif tag in ('связь','рой/связь'): tt=random.uniform(-1,1); x=tt; y=0.85*(1-tt*tt)-0.35+G(0.05); z=G(0.12)
        elif tag=='спираль': tt=random.random(); a=tt*6.283*1.6; x=math.cos(a)*0.65; y=tt*2-1; z=math.sin(a)*0.65
        elif tag=='покой': x=random.uniform(-0.95,0.95); z=random.uniform(-0.6,0.6); y=G(0.07)
        elif tag in ('ручей','ветер'): x=random.uniform(-1.05,1.05); y=G(0.2); z=random.uniform(-0.4,0.4)
        elif tag=='ширь': x=random.uniform(-1.25,1.25); y=G(0.05); z=random.uniform(-0.32,0.32)
        elif tag=='пересечение': d=random.choice([1,-1]); tt=random.uniform(-1,1); x=tt; y=tt*d; z=G(0.1)
        elif tag in ('касание','становление'): s=random.choice([-0.5,0.5]); x=s+G(0.22); y=G(0.3); z=G(0.22)
        else: x=G(0.45); y=G(0.45); z=G(0.4)
        p.append((x,y,z))
    return p
def hsl(h,s,l):
    import colorsys; r,g,b=colorsys.hls_to_rgb((h%360)/360,l,s); return int(r*255),int(g*255),int(b*255)

def main():
    карта=json.load(open(КАРТА)); by={b['буква']:b for b in карта['буквы']}
    найдено=[]
    for буква,tr in TR.items():
        srcs=glob.glob(os.path.join(ВХОД,tr+'.*'))
        srcs=[s for s in srcs if s.rsplit('.',1)[-1].lower() in ('wav','mp3','m4a','flac','aac','ogg')]
        if not srcs: continue
        src=srcs[0]; b=by.get(буква)
        if not b: continue
        random.seed(ord(буква)); np.random.seed(ord(буква)%2000)
        x,real=decode(src,tr); pool=атомизировать(x)
        if not pool: print(f"  {буква}: запись слишком тихая/короткая — пропуск"); continue
        T,H=фаза_кадры(x); pr=params104(x); b['параметры_звука']=pr
        sat=0.4+pr['harmonicity']*0.5; PER=1500; pts=форма(b['динамика'],PER); n=len(pts); atoms=[]
        for i,q in enumerate(pts):
            r,g,bl=hsl(b['цвет_hue'],sat,0.46+random.random()*0.22)
            pa=pool[np.random.randint(len(pool))]; bt=i/n*1.0; idx=min(int(np.searchsorted(T,bt)),len(H)-1)
            atoms.append({"birth":round(bt,4),"freq":round(float(pa[1]),2),"amp":round(float(pa[2]),4),
                "harmonic_index":1,"harmonicity":round(float(pa[3]),3),"phase":ph(float(H[idx])),
                "color_r":r,"color_g":g,"color_b":bl,"pos_x":round(q[0]*0.9,3),"pos_y":round(q[1],3),"pos_z":round(q[2]*0.9,3)})
        json.dump({"atoms":atoms},open(os.path.join(ET,f"bukva_{tr}_model3d.json"),"w"),ensure_ascii=False)
        from collections import Counter
        dom=Counter(a['phase'] for a in atoms).most_common(1)[0][0]
        найдено.append((буква,dom,pr['centroid'])); print(f"  {буква} ({tr}): живой голос → фаза {dom}, centroid {pr['centroid']}, форманты {pr['formantF1']}/{pr['formantF2']}")
    json.dump(карта,open(КАРТА,'w'),ensure_ascii=False,indent=2)
    print(f"\nпересобрано живых букв: {len(найдено)} из 33")
    if найдено: print("Дальше: пересчитать п104 (cell_params) и обновить буквица_карта (уже сделано).")

if __name__=='__main__': main()

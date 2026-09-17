// Тринити Альфа — проекция «ОБРАЗ»: НАСТОЯЩЕЕ ЕДИНСТВО сцены и атомов.
// Сцена строится ИЗ ТЕХ ЖЕ атомов, целиком по проверенному алфавиту:
//   позиция ← частота/время/амплитуда, цвет ← частота (Hz),
//   насыщенность ← гармоничность (тембр), размер ← амплитуда.
// Никаких рецептов/iframe. Образ разворачивается во времени (birth) и ЖИВЁТ —
// пульсирует от живого спектра (window.__плеер.analyser). Сцена ≡ атомы.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { создать_пульс_вложенности } from './вложенность_пульс.js';

// ── буквы алфавита ──
function freqToHue(hz){ const t=Math.log10(Math.max(20,Math.min(hz,20000))/20)/Math.log10(1000); return 270-Math.min(t,1)*270; }
function hslToRgb(h,s,l){ h/=360; const a=s*Math.min(l,1-l); const f=k=>{const x=(k+h*12)%12; return l-a*Math.max(-1,Math.min(x-3,9-x,1));}; return [f(0),f(8),f(4)]; }
function норм(v,a,b){ return b===a?0.5:(v-a)/(b-a); }

// круглый светящийся спрайт точки
function glowTexture(){
    const c=document.createElement('canvas'); c.width=c.height=64; const g=c.getContext('2d');
    const grad=g.createRadialGradient(32,32,0,32,32,32);
    grad.addColorStop(0,'rgba(255,255,255,1)'); grad.addColorStop(0.35,'rgba(255,255,255,0.55)');
    grad.addColorStop(1,'rgba(255,255,255,0)');
    g.fillStyle=grad; g.fillRect(0,0,64,64);
    const tex=new THREE.CanvasTexture(c); tex.needsUpdate=true; return tex;
}

export function создать_образ_среду(контейнер_родитель){
    const узел=document.createElement('div'); узел.className='проекция проекция-образ'; контейнер_родитель.appendChild(узел);
    const подсказка=document.createElement('div'); подсказка.className='подсказка-вращения';
    подсказка.textContent='Образ из атомов · цвет=Hz · насыщенность=тембр · оживает при Play'; узел.appendChild(подсказка);
    const индикатор=document.createElement('div'); индикатор.className='индикатор-загрузки'; индикатор.hidden=true; узел.appendChild(индикатор);

    const сцена=new THREE.Scene(); сцена.background=new THREE.Color(0x000000);
    const камера=new THREE.PerspectiveCamera(50,1,0.1,1000); камера.position.set(0,0,6);
    const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(window.devicePixelRatio,2)); узел.appendChild(renderer.domElement);
    const controls=new OrbitControls(камера,renderer.domElement); controls.enableDamping=true; controls.dampingFactor=0.08;
    const группа=new THREE.Group(); сцена.add(группа);
    const tex=glowTexture();

    let облако=null, geom=null;
    let рёбра_группа=null, crossSegs=[], crossIdxMap=null;
    let freqs=null, baseSize=null, births=null, baseY=null, baseX=null, phases=null;
    // шов 4: локальные 28 полей атома → живость каждой точки
    let noiseL=null, ampModL=null, timbreModL=null, dissL=null, envPh=null, baseCol=null;
    let динамика='', yMin=0, yRange=1, xMin=0, xRange=1, былПлей=false, шерох=0, _пульс=null;
    let jitterK=0, shimmerK=0, fdK=0;   // шов 5: жизнь клетки из 104
    let последнее_t=Infinity, время0=0;

    function pick(...vals){ for(const v of vals){ if(v!=null && !Number.isNaN(+v)) return +v; } return null; }

    function resize(){ const r=узел.getBoundingClientRect(); const w=Math.max(1,Math.round(r.width)),h=Math.max(1,Math.round(r.height));
        renderer.setSize(w,h,false); камера.aspect=w/h; камера.updateProjectionMatrix(); }
    const ro=new ResizeObserver(resize); ro.observe(узел); resize();

    let активна=false, raf=0;
    function tick(){
        if(!активна) return;
        controls.update();
        // ЖИВОСТЬ: пульс от живого спектра
        const пл=window.__плеер;
        const играет = облако && freqs && пл && пл.analyser && пл.аудио && !пл.аудио.paused;
        if(играет){
            былПлей=true;
            const an=пл.analyser; const bins=an.frequencyBinCount;
            if(!tick._spec || tick._spec.length!==bins) tick._spec=new Uint8Array(bins);
            an.getByteFrequencyData(tick._spec);
            const ny=(пл.audioCtx?.sampleRate||44100)/2;
            const sizeAttr=geom.getAttribute('size'); const posAttr=geom.getAttribute('position');
            const colAttr=geom.getAttribute('color');
            const N=облако.geometry.drawRange.count; const t=performance.now();
            let L=0; for(let b=0;b<bins;b++) L+=tick._spec[b]; L=L/bins/255;   // общая громкость (для грома: вспышка+волна)
            // ── НАПРАВЛЕНИЕ от явления (поле «КУДА»). Характер «КАК» — ниже, из фазы. ──
            const падёт=(динамика==='дождь'||динамика==='водопад'||динамика==='песок');
            const vY=(динамика==='водопад')?1.8:(динамика==='песок'?1.3:1.0);
            const offY=падёт ? (t*0.00060*vY)%1 : 0;
            const течётX=(динамика==='ветер'||динамика==='ручей');
            const offX=течётX ? (t*0.00045*(динамика==='ручей'?1.3:1.0))%1 : 0;
            const вверх=(динамика==='огонь'); const offU=вверх ? (t*0.00040)%1 : 0;
            const волна=(динамика==='волны');
            const радиал=(динамика==='гром'||динамика==='удар'||динамика==='пульс'||динамика==='дыхание');
            const kR=радиал ? (1+L*((динамика==='удар')?0.5:(динамика==='пульс')?0.45:0.35)) : 1;
            const вспых=(динамика==='удар')?5.0:(динамика==='гром')?4.0:0.0;          // импульсные явления — вспышка размера
            for(let i=0;i<N;i++){
                const bin=Math.min(bins-1,Math.max(0,Math.floor((freqs[i]/ny)*bins)));
                const ex=tick._spec[bin]/255;
                let px=baseX[i], py=baseY[i]; const pz=posAttr.array[i*3+2];
                // НАПРАВЛЕНИЕ (явление)
                if(падёт){ py=baseY[i]-offY*yRange; if(py<yMin) py+=yRange; }
                else if(течётX){ px=baseX[i]+offX*xRange; if(px>xMin+xRange) px-=xRange; }
                else if(вверх){ py=baseY[i]+offU*yRange; if(py>yMin+yRange) py-=yRange; }
                else if(волна){ py=baseY[i]+(0.18+L*0.5)*Math.sin(baseX[i]*1.3 - t*0.0022 + pz*0.5); }
                else if(радиал){ px=baseX[i]*kR; py=baseY[i]*kR; }
                // ХАРАКТЕР (фаза): тон→жёсткий покой, шум→диффузный дрейф, переход→мерцание
                const ph=phases?phases[i]:''; let sizeK;
                if(ph==='тон'){ sizeK=1+ex*0.6; }                                                  // кристалл — без дрожи
                else if(ph==='шум'){ sizeK=1+ex*1.1; px+=Math.sin(t*0.0011+i*1.3)*0.05; py+=Math.cos(t*0.0009+i*2.1)*0.05; } // дымка дрейфует
                else if(ph==='переход'){ sizeK=1+ex*1.4; px+=Math.sin(t*0.0026+i*1.7)*0.06; py+=Math.cos(t*0.0023+i*2.3)*0.06; } // мерцание
                else { sizeK=1+ex*1.2; py+=ex*0.02*Math.sin(t*0.003+i*7.13); }                     // нет фазы — лёгкое дыхание
                if(шерох>0.08){ px+=шерох*0.04*Math.sin(t*0.018+i*3.7); py+=шерох*0.04*Math.cos(t*0.016+i*5.1); }
                // шов 4–5: локальная + клеточная жизнь
                const nl=noiseL?noiseL[i]:0, am=ampModL?ampModL[i]:0, tm=timbreModL?timbreModL[i]:0, ds=dissL?dissL[i]:0;
                if(nl>0.02 || jitterK>0){ const j=(nl*0.06+jitterK*0.03); px+=j*Math.sin(t*0.022+i*4.1); py+=j*Math.cos(t*0.019+i*3.3); }
                if(ds>0.05){ px+=ds*0.05*Math.sin(i*2.7); py+=ds*0.05*Math.cos(i*3.1); pz+=ds*0.03*Math.sin(i*1.9); }
                if(tm && baseCol && colAttr){ const dr=tm*0.08*Math.sin(t*0.0015+i); colAttr.array[i*3]=Math.min(1,baseCol[i*3]+dr); colAttr.array[i*3+2]=Math.min(1,baseCol[i*3+2]-dr*0.5); }
                posAttr.array[i*3]=px; posAttr.array[i*3+1]=py; posAttr.array[i*3+2]=pz;
                const pk=_пульс?_пульс.пульс(freqs[i],t):1;
                const sh=(shimmerK>0?1+shimmerK*0.25*Math.sin(t*0.014+i*2.9):1);
                const amF=(am>0?1+am*0.35*Math.sin(t*0.017+i*5.7):1);
                sizeAttr.array[i]=baseSize[i]*(sizeK + L*вспых)*pk*sh*amF;
            }
            sizeAttr.needsUpdate=true; posAttr.needsUpdate=true;
            if(colAttr) colAttr.needsUpdate=true;
        } else if(былПлей){                                                          // пауза/стоп — вернуть чистую форму
            былПлей=false;
            const sizeAttr=geom.getAttribute('size'); const posAttr=geom.getAttribute('position');
            const N=облако?облако.geometry.drawRange.count:0;
            for(let i=0;i<N;i++){ posAttr.array[i*3]=baseX[i]; posAttr.array[i*3+1]=baseY[i]; sizeAttr.array[i]=baseSize[i]; }
            if(N){ sizeAttr.needsUpdate=true; posAttr.needsUpdate=true; }
        }
        renderer.render(сцена,камера);
        raf=requestAnimationFrame(tick);
    }

    function вписать(){ const box=new THREE.Box3().setFromObject(группа); if(box.isEmpty())return;
        const c=box.getCenter(new THREE.Vector3()); const s=box.getSize(new THREE.Vector3()).length()||1;
        controls.target.copy(c); камера.position.copy(c).add(new THREE.Vector3(0,0,s*1.3));
        камера.near=s/100; камера.far=s*100; камера.updateProjectionMatrix(); controls.update(); }

    // ОБРАЗ из атомов (одна логика — алфавит)
    function применить(данные, режим_коорд='радиус', цвет_режим='файл', дин='', п104=null, оси104=null){
        динамика=дин||'';
        const O=оси104||{};
        // ВЛОЖЕННОСТЬ → синхронная пульсация (из паспорта 104). См. вложенность_пульс.js
        _пульс=создать_пульс_вложенности({nestedness:O.nestedness, mod_depth:O.mod_depth, mod_rate:O.mod_rate});
        // шов 3–5: калибровка + жизнь (п104 — защита; n_* из 104 — fallback)
        jitterK=Math.min(1, pick(O.jitter) || 0);
        shimmerK=Math.min(1, pick(O.shimmer) || 0);
        fdK=Math.min(1, pick(O.fd) || 0);
        const хвост=pick(п104?.n_хвост, O.n_хвост, O.ring_decay) ?? 0;
        const тепло=pick(п104?.n_тепло, O.n_тепло, O.perceptual_warmth) ?? 0.5;
        шерох=pick(п104?.n_шероховатость, O.n_шероховатость, O.roughness, fdK) ?? 0;
        const спред=1+хвост*0.10, свечение=1+хвост*0.5, wt=(тепло-0.5);
        while(группа.children.length){ const o=группа.children.pop(); o.geometry?.dispose?.(); o.material?.dispose?.(); }
        облако=null; geom=null; freqs=null; crossSegs=[]; рёбра_группа=null;
        const исх=данные?.atoms||[]; const n=исх.length; if(!n) return false;
        const idx=Array.from({length:n},(_,i)=>i).sort((i,j)=>(исх[i].birth??0)-(исх[j].birth??0));
        const origToSorted={}; for(let j=0;j<n;j++) origToSorted[idx[j]]=j;
        const A=idx.map(i=>исх[i]);
        for(const a of A){
            const p=a.params_104||{};
            if(p.jitter!=null) jitterK=Math.max(jitterK, Math.min(1, +p.jitter));
            if(p.shimmer!=null) shimmerK=Math.max(shimmerK, Math.min(1, +p.shimmer));
            if(p.fd!=null) fdK=Math.max(fdK, Math.min(1, +p.fd));
            if(p.roughness!=null && шерох<0.02) шерох=Math.min(1, +p.roughness);
        }
        const tmin=Math.min(...A.map(a=>a.birth||0)), tmax=Math.max(...A.map(a=>a.birth||0));
        const lf=A.map(a=>Math.log10(Math.max(1,a.freq||20))); const fmin=Math.min(...lf),fmax=Math.max(...lf);
        const amax=Math.max(1e-6,...A.map(a=>a.amp||0));
        const pos=new Float32Array(n*3), col=new Float32Array(n*3), sz=new Float32Array(n), bf=new Float32Array(n);
        freqs=new Float32Array(n); baseSize=new Float32Array(n); births=new Float32Array(n); baseY=new Float32Array(n); baseX=new Float32Array(n); phases=new Array(n);
        noiseL=new Float32Array(n); ampModL=new Float32Array(n); timbreModL=new Float32Array(n); dissL=new Float32Array(n); envPh=new Array(n);
        baseCol=new Float32Array(n*3);
        for(let i=0;i<n;i++){
            const a=A[i]; const ampN=Math.min(1,(a.amp||0)*6); const f=a.freq||20;
            const p104=a.params_104||{};
            noiseL[i]=Math.min(1, +(a.noisiness_local||p104.spectral_flatness||0));
            ampModL[i]=Math.min(1, Math.abs(+(a.amp_modulation||0)));
            timbreModL[i]=Math.min(1, Math.abs(+(a.timbre_modulation||0)));
            dissL[i]=Math.min(1, +(a.dissonance_local||0));
            envPh[i]=a.envelope_phase||'';
            let x,y,z;
            if(a.pos_x!==undefined && режим_коорд!=='радиус' && режим_коорд!=='физика'){
                // ЭТАЛОН/Авто: позиции из картинки — образ сохраняет реальную форму
                x=+a.pos_x||0; y=+a.pos_y||0; z=+a.pos_z||0;
            } else if(режим_коорд==='радиус'){ // радиус — волна во все стороны (по запросу)
                const угол=i*2.399963229; const радиус=0.3+2.6*норм(a.birth||0,tmin,tmax);
                x=Math.cos(угол)*радиус; z=Math.sin(угол)*радиус;
                y=(норм(Math.log10(Math.max(1,f)),fmin,fmax)-0.5)*2.0;
            } else { // ПО УМОЛЧАНИЮ — физика: X=время, Y=log частоты, Z=амплитуда (структура звука видна)
                x=(2*норм(a.birth||0,tmin,tmax)-1)*1.8;
                y=(норм(Math.log10(Math.max(1,f)),fmin,fmax)-0.5)*2.4;
                z=((a.amp||0)/amax)*1.2;
            }
            // ГРАММАТИКА ГИББСА: фаза атома задаёт фактуру (тон→кристалл/решётка, шум→дымка/рассеяние, переход→мерцание в tick)
            const ph=a.phase||''; phases[i]=ph;
            if(ph==='тон'){ const st=0.10; x=Math.round(x/st)*st; y=Math.round(y/st)*st; z=Math.round(z/st)*st; }      // лёгкая кристаллизация
            else if(ph==='шум'){ x+=(Math.random()-.5)*0.07; y+=(Math.random()-.5)*0.07; z+=(Math.random()-.5)*0.07; }
            // шов 4: локальные признаки → форма атома
            if(dissL[i]>0.03){ x+=dissL[i]*0.04*(Math.random()-.5); y+=dissL[i]*0.04*(Math.random()-.5); z+=dissL[i]*0.03*(Math.random()-.5); }
            if(envPh[i]==='attack'){ const k=0.85; x*=k; y*=k; z*=k; }
            else if(envPh[i]==='decay'||envPh[i]==='release'){ x*=1.02; y*=1.02; }
            x*=спред; y*=спред; z*=спред;
            pos[i*3]=x; pos[i*3+1]=y; pos[i*3+2]=z; baseY[i]=y; baseX[i]=x;
            let r,g,b;
            if(цвет_режим==='файл' && a.color_r!=null){   // ЦВЕТ ИЗ ФОРМЫ — образ выглядит как явление (огонь тёплый, дождь синий)
                r=(+a.color_r)/255; g=(+a.color_g)/255; b=(+a.color_b)/255;
            } else {                                       // ЦВЕТ = Hz (спектральный образ)
                const harm=(a.harmonicity!=null)?a.harmonicity:0.6;   // тембр → насыщенность
                // КАЛИБРОВКА: тон↔шум бимодален (щель Гиббса) — гамма-развёртка раскрывает середину
                [r,g,b]=hslToRgb(freqToHue(f), 0.12+Math.pow(harm,0.4)*0.85, 0.45+ampN*0.4);
            }
            if(ph==='шум'){ r=r*0.82+0.10; g=g*0.82+0.11; b=b*0.82+0.14; }   // дымка — лёгкое осветление, цвет явления сохраняется (дождь синий)
            r=Math.max(0,Math.min(1,r+wt*0.12)); b=Math.max(0,Math.min(1,b-wt*0.12));  // наклон → тёплый/холодный сдвиг
            col[i*3]=r; col[i*3+1]=g; col[i*3+2]=b;
            baseCol[i*3]=r; baseCol[i*3+1]=g; baseCol[i*3+2]=b;
            const s=(0.025+ampN*0.075)*(1+ampModL[i]*0.15)*свечение; sz[i]=s; baseSize[i]=s;
            bf[i]=a.birth||0; births[i]=a.birth||0; freqs[i]=f;
        }
        let ylo=Infinity,yhi=-Infinity,xlo=Infinity,xhi=-Infinity;
        for(let i=0;i<n;i++){ if(baseY[i]<ylo)ylo=baseY[i]; if(baseY[i]>yhi)yhi=baseY[i]; if(baseX[i]<xlo)xlo=baseX[i]; if(baseX[i]>xhi)xhi=baseX[i]; }
        yMin=ylo; yRange=Math.max(0.001,yhi-ylo); xMin=xlo; xRange=Math.max(0.001,xhi-xlo);
        geom=new THREE.BufferGeometry();
        geom.setAttribute('position',new THREE.BufferAttribute(pos,3));
        geom.setAttribute('color',new THREE.BufferAttribute(col,3));
        geom.setAttribute('size',new THREE.BufferAttribute(sz,1));
        // Светящиеся точки с per-atom размером (ShaderMaterial). Это и есть «сцена» из атомов.
        const mat=new THREE.ShaderMaterial({
            uniforms:{ uTex:{ value:tex } },
            vertexShader:
                'attribute float size;\n'+
                'attribute vec3 color;\n'+
                'varying vec3 vColor;\n'+
                'void main(){ vColor=color; vec4 mv=modelViewMatrix*vec4(position,1.0);'+
                ' gl_PointSize=size*(200.0/-mv.z); gl_Position=projectionMatrix*mv; }',
            fragmentShader:
                'uniform sampler2D uTex;\n'+
                'varying vec3 vColor;\n'+
                'void main(){ vec4 t=texture2D(uTex,gl_PointCoord); if(t.a<0.02) discard;'+
                ' gl_FragColor=vec4(vColor,1.0)*t; }',
            transparent:true, depthWrite:false, blending:THREE.AdditiveBlending,
        });
        облако=new THREE.Points(geom,mat); группа.add(облако);
        geom.setDrawRange(0,n);
        const crosses=данные?.crosses;
        if(crosses?.length) построить_кресты_образ(crosses, origToSorted, pos, bf);
        вписать(); setVisibleByTime(Infinity);
        return true;
    }

    function upperBound(arr,t){ if(!arr||!arr.length)return 0; if(!isFinite(t))return arr.length;
        let lo=0,hi=arr.length; while(lo<hi){const m=(lo+hi)>>>1; if(arr[m]<=t)lo=m+1; else hi=m;} return lo; }

    const ЦВЕТ_ОСИ={ time:0x4682B4, freq:0x32CD32, harmonic:0xFF6347 };

    function построить_кресты_образ(crosses, origToSorted, posArr, birthsArr){
        crossSegs=[]; if(!crosses?.length||!origToSorted) return;
        рёбра_группа=new THREE.Group();
        const byAxis={};
        for(const c of crosses){
            const sa=origToSorted[c.atom_a], sb=origToSorted[c.atom_b];
            if(sa==null||sb==null) continue;
            const ax=c.axis||'time';
            if(!byAxis[ax]) byAxis[ax]=[];
            byAxis[ax].push({ sa, sb, th:Math.max(birthsArr[sa]||0, birthsArr[sb]||0) });
        }
        for(const [ax, pairs] of Object.entries(byAxis)){
            pairs.sort((a,b)=>a.th-b.th);
            const verts=new Float32Array(pairs.length*6);
            const thresholds=new Float32Array(pairs.length);
            for(let k=0;k<pairs.length;k++){
                const {sa,sb,th}=pairs[k];
                verts[k*6  ]=posArr[sa*3  ]; verts[k*6+1]=posArr[sa*3+1]; verts[k*6+2]=posArr[sa*3+2];
                verts[k*6+3]=posArr[sb*3  ]; verts[k*6+4]=posArr[sb*3+1]; verts[k*6+5]=posArr[sb*3+2];
                thresholds[k]=th;
            }
            const g=new THREE.BufferGeometry();
            g.setAttribute('position',new THREE.BufferAttribute(verts,3));
            g.setDrawRange(0, pairs.length*2);
            const mat=new THREE.LineBasicMaterial({
                color:ЦВЕТ_ОСИ[ax]||0x888888, transparent:true, opacity:0.28, depthWrite:false,
            });
            const seg=new THREE.LineSegments(g,mat);
            seg.userData.ось=ax;
            crossSegs.push({ seg, thresholds, count:pairs.length });
            рёбра_группа.add(seg);
        }
        if(рёбра_группа.children.length) группа.add(рёбра_группа);
    }

    function setCrossVisibleByTime(t){
        if(!crossSegs.length) return;
        for(const { seg, thresholds, count } of crossSegs){
            let vis=0;
            if(!isFinite(t)) vis=count;
            else for(let k=0;k<count;k++) if(thresholds[k]<=t) vis=k+1;
            seg.geometry.setDrawRange(0, vis*2);
        }
    }
    function setVisibleByTime(t){ последнее_t=t; if(облако&&births){ const N=upperBound(births,t); облако.geometry.setDrawRange(0,N);} setCrossVisibleByTime(t); }

    return {
        узел,
        активировать(){ узел.classList.add('активна'); активна=true; resize(); raf=requestAnimationFrame(tick); },
        деактивировать(){ узел.classList.remove('активна'); активна=false; cancelAnimationFrame(raf); },
        применить, setVisibleByTime,
        показать_индикатор(t){ индикатор.textContent=t||'…'; индикатор.hidden=false; },
        спрятать_индикатор(){ индикатор.hidden=true; },
        снять(){ ro.disconnect(); cancelAnimationFrame(raf); controls.dispose(); renderer.dispose(); узел.remove(); },
    };
}

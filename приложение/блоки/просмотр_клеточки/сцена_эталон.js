// Тринити Альфа — ЭТАЛОННАЯ 3D-СЦЕНА (свой рендер, не iframe).
// Чистая 3D-модель явления, из неё формируются атомы (Сцена ≡ Атомы ≡ Звук).
// Типы: «дождь» (падающие струи-линии), «огонь» (восходящие тёплые частицы).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function glowTex(){
    const c=document.createElement('canvas'); c.width=c.height=64; const g=c.getContext('2d');
    const gr=g.createRadialGradient(32,32,0,32,32,32);
    gr.addColorStop(0,'rgba(255,255,255,1)'); gr.addColorStop(0.4,'rgba(255,255,255,0.5)'); gr.addColorStop(1,'rgba(255,255,255,0)');
    g.fillStyle=gr; g.fillRect(0,0,64,64); const t=new THREE.CanvasTexture(c); t.needsUpdate=true; return t;
}
function warm(yf,b){ // огонь: низ красный → верх жёлтый
    const hue=(12+yf*48)/360, l=0.44+yf*0.42*b, s=0.95, a=s*Math.min(l,1-l);
    const f=k=>{const x=(k+hue*12)%12; return l-a*Math.max(-1,Math.min(x-3,9-x,1));};
    return [f(0),f(8),f(4)];
}

export function создать_эталон_сцену(корень_родитель){
    const узел=document.createElement('div'); узел.className='проекция проекция-образ эталон-сцена'; корень_родитель.appendChild(узел);
    const подпись=document.createElement('div'); подпись.className='подсказка-вращения';
    подпись.textContent='3D-эталон явления · из него формируются атомы'; узел.appendChild(подпись);

    const сцена=new THREE.Scene(); сцена.background=new THREE.Color(0x000000);
    const камера=new THREE.PerspectiveCamera(50,1,0.1,1000); камера.position.set(0,0,7);
    const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(window.devicePixelRatio,2)); узел.appendChild(renderer.domElement);
    const controls=new OrbitControls(камера,renderer.domElement); controls.enableDamping=true; controls.dampingFactor=0.08;
    const группа=new THREE.Group(); сцена.add(группа);
    const tex=glowTex();

    let model=null, segs=null, pts=null, wind=null, t0=performance.now(), уровень=0;
    function resize(){ const r=узел.getBoundingClientRect(); const w=Math.max(1,Math.round(r.width)),h=Math.max(1,Math.round(r.height));
        renderer.setSize(w,h,false); камера.aspect=w/h; камера.updateProjectionMatrix(); }
    const ro=new ResizeObserver(resize); ro.observe(узел); resize();

    let активна=false, raf=0;
    function tick(){
        if(!активна) return; controls.update();
        const dt=(performance.now()-t0)/1000;
        // СИНХРОНИЗАЦИЯ С ЖИВЫМ ЗВУКОМ: сцена реагирует на играющий звук (как Образ)
        let L=0; { const пл=window.__плеер;
            if(пл && пл.analyser && пл.аудио && !пл.аудио.paused){
                const an=пл.analyser, bins=an.frequencyBinCount;
                if(!tick._spec||tick._spec.length!==bins) tick._spec=new Uint8Array(bins);
                an.getByteFrequencyData(tick._spec);
                let s=0; for(let b=0;b<bins;b++) s+=tick._spec[b]; L=s/bins/255;
            } }
        уровень += (L-уровень)*0.35;                       // сглаженная громкость
        const живой = 0.30 + уровень*1.6;                  // множитель живости (база — чтобы видно на паузе)
        if(segs && model && model.drops){            // ДОЖДЬ/ВОДОПАД — падающие линии
            const span=model.y_top-model.y_bot, pos=segs.geometry.attributes.position.array, D=model.drops;
            for(let i=0;i<D.length;i++){
                let p=(D[i].phase + dt*model.скорость)%1; if(p<0)p+=1;
                const head=model.y_top - p*span, tail=head-D[i].len;
                pos[i*6]=D[i].x; pos[i*6+1]=head; pos[i*6+2]=D[i].z;
                pos[i*6+3]=D[i].x; pos[i*6+4]=tail; pos[i*6+5]=D[i].z;
            }
            segs.geometry.attributes.position.needsUpdate=true;
            segs.material.opacity=Math.min(1,0.30+уровень*0.9);    // яркость струй в такт звуку
        }
        if(wind && model && model.тип==='ветер' && model.parts){   // ВЕТЕР — горизонтальный поток струй
            const span=model.x_right-model.x_left, pos=wind.geometry.attributes.position.array, W=model.parts;
            for(let i=0;i<W.length;i++){
                let p=(W[i].phase + dt*model.скорость)%1; if(p<0)p+=1;
                const head=model.x_left + p*span, tail=head-W[i].len;
                pos[i*6]=head;   pos[i*6+1]=W[i].y; pos[i*6+2]=W[i].z;
                pos[i*6+3]=tail;  pos[i*6+4]=W[i].y; pos[i*6+5]=W[i].z;
            }
            wind.geometry.attributes.position.needsUpdate=true;
            wind.material.opacity=Math.min(1,0.30+уровень*0.85);   // порывы ветра в такт звуку
        }
        if(pts && model && model.parts){             // ОГОНЬ — восходящие частицы
            const P=model.parts, posA=pts.geometry.attributes.position.array, colA=pts.geometry.attributes.color.array, szA=pts.geometry.attributes.size.array;
            const ybot=model.y_bot;
            for(let i=0;i<P.length;i++){
                let p=(P[i].phase + dt*model.скорость)%1; if(p<0)p+=1;
                const yf=p, y=ybot + yf*P[i].h, narrow=1-yf*0.7;
                posA[i*3]=P[i].x*narrow; posA[i*3+1]=y; posA[i*3+2]=P[i].z*narrow;
                const [r,g,b]=warm(yf,P[i].bright); colA[i*3]=r; colA[i*3+1]=g; colA[i*3+2]=b;
                szA[i]=(0.08+P[i].bright*0.12)*(1-yf*0.35)*(0.6+уровень*1.4);   // мерцание в такт треску
            }
            pts.geometry.attributes.position.needsUpdate=true;
            pts.geometry.attributes.color.needsUpdate=true; pts.geometry.attributes.size.needsUpdate=true;
        }
        if(pts && model && model.тип==='волны' && model.точки){   // ВОЛНЫ — бегущая зыбь + пена на гребне
            const P=model.точки, posA=pts.geometry.attributes.position.array, szA=pts.geometry.attributes.size.array, colA=pts.geometry.attributes.color.array;
            const A=(model.амплитуда||0.4)*(0.5+уровень*1.3), k=model.длина_волны||1.2, sp=model.скорость||1.0, yb=model.y_base||0;  // вздымание в такт прибою
            for(let i=0;i<P.length;i++){
                const s=Math.sin(P[i].x*k - dt*sp + P[i].z*0.4);
                posA[i*3]=P[i].x; posA[i*3+1]=yb + A*s; posA[i*3+2]=P[i].z;
                const foam=Math.max(0,(s-0.6)/0.4);                        // гребень → пена
                colA[i*3]=0.30+foam*0.70; colA[i*3+1]=0.55+foam*0.45; colA[i*3+2]=0.85+foam*0.15;
                szA[i]=0.06+foam*0.13;
            }
            pts.geometry.attributes.position.needsUpdate=true; pts.geometry.attributes.size.needsUpdate=true; pts.geometry.attributes.color.needsUpdate=true;
        }
        if(model && model.тип==='гром'){   // ГРОМ — вспышка молнии + затухание + рокот облака
            const env=Math.min(1,0.05+уровень*3.2);   // вспышка идёт от РЕАЛЬНОГО раската (громкости), а не по таймеру
            const мерц=0.55+0.45*Math.sin(dt*42+Math.sin(dt*6.0));   // дрожание молнии
            if(segs) segs.material.opacity=Math.min(1,0.06+env*мерц);
            if(pts && model.частицы){ const P=model.частицы, posA=pts.geometry.attributes.position.array, szA=pts.geometry.attributes.size.array;
                for(let i=0;i<P.length;i++){ const j=P[i].phase*6.283;
                    posA[i*3]=P[i].x+Math.sin(dt*1.5+j)*0.05; posA[i*3+1]=P[i].y+Math.cos(dt*1.3+j)*0.05; posA[i*3+2]=P[i].z;
                    szA[i]=(0.06+(P[i].bright||0.6)*0.08)*(0.5+env*2.2); }
                pts.geometry.attributes.position.needsUpdate=true; pts.geometry.attributes.size.needsUpdate=true;
            }
        }
        if(pts && model && (model.тип==='пульс'||model.тип==='дыхание') && model.частицы){   // ПУЛЬС/ДЫХАНИЕ — масштаб от звука
            const P=model.частицы, posA=pts.geometry.attributes.position.array, szA=pts.geometry.attributes.size.array;
            const пульс=model.тип==='пульс', k=1+уровень*(пульс?0.6:0.4);   // сердце бьётся резче, дыхание плавнее
            for(let i=0;i<P.length;i++){ posA[i*3]=P[i].x*k; posA[i*3+1]=P[i].y*k; posA[i*3+2]=P[i].z*k;
                szA[i]=(0.07+(P[i].bright||0.8)*0.07)*(0.6+уровень*(пульс?2.2:1.3)); }
            pts.geometry.attributes.position.needsUpdate=true; pts.geometry.attributes.size.needsUpdate=true;
        }
        if(pts && model && model.тип==='удар' && model.частицы){   // УДАР — вспышка+подскок от реального удара
            const P=model.частицы, posA=pts.geometry.attributes.position.array, szA=pts.geometry.attributes.size.array;
            for(let i=0;i<P.length;i++){ const j=P[i].phase*6.283;
                posA[i*3]=P[i].x; posA[i*3+1]=P[i].y + уровень*0.5*Math.abs(Math.sin(dt*3+j)); posA[i*3+2]=P[i].z;
                szA[i]=(0.06+(P[i].bright||0.7)*0.07)*(0.5+уровень*3.0); }   // вспышка на ударе
            pts.geometry.attributes.position.needsUpdate=true; pts.geometry.attributes.size.needsUpdate=true;
        }
        if(wind && model && model.тип==='ручей' && model.parts){   // РУЧЕЙ — поток струй вправо
            const span=model.x_right-model.x_left, pos=wind.geometry.attributes.position.array, W=model.parts;
            for(let i=0;i<W.length;i++){
                let p=(W[i].phase + dt*model.скорость)%1; if(p<0)p+=1;
                const head=model.x_left + p*span, tail=head-W[i].len;
                pos[i*6]=head; pos[i*6+1]=W[i].y; pos[i*6+2]=W[i].z;
                pos[i*6+3]=tail; pos[i*6+4]=W[i].y; pos[i*6+5]=W[i].z;
            }
            wind.geometry.attributes.position.needsUpdate=true;
            wind.material.opacity=Math.min(1,0.30+уровень*0.85);   // журчание ярче в такт звуку
        }
        if(pts && model && model.тип==='ручей' && model.bubbles){   // РУЧЕЙ — пузырьки всплывают/булькают
            const M=model.bubbles, posA=pts.geometry.attributes.position.array, szA=pts.geometry.attributes.size.array;
            for(let i=0;i<M.length;i++){ const j=M[i].phase*6.283;
                posA[i*3]=M[i].x + Math.sin(dt*0.8+j)*0.08;
                posA[i*3+1]=M[i].y + Math.abs(Math.sin(dt*1.6+j))*0.12;
                posA[i*3+2]=M[i].z;
                szA[i]=(0.05+(M[i].bright||0.7)*0.06)*(0.6+0.4*Math.sin(dt*3+i))*(0.6+уровень*1.2); }
            pts.geometry.attributes.position.needsUpdate=true; pts.geometry.attributes.size.needsUpdate=true;
        }
        if(pts && model && model.тип==='водопад' && model.mist){   // ВОДОПАД — туман/брызги внизу
            const M=model.mist, posA=pts.geometry.attributes.position.array, szA=pts.geometry.attributes.size.array;
            for(let i=0;i<M.length;i++){
                const ph=M[i].phase*6.283;
                posA[i*3]=M[i].x + Math.sin(dt*0.7+ph)*0.10;
                posA[i*3+1]=model.y_bot + 0.15 + Math.abs(Math.sin(dt*0.9+ph))*0.45;   // клубится у подножия
                posA[i*3+2]=M[i].z + Math.cos(dt*0.6+ph)*0.10;
                szA[i]=(0.06+M[i].bright*0.07)*(0.7+0.3*Math.sin(dt*2.0+i))*(0.6+уровень*1.2);
            }
            pts.geometry.attributes.position.needsUpdate=true; pts.geometry.attributes.size.needsUpdate=true;
        }
        renderer.render(сцена,камера); raf=requestAnimationFrame(tick);
    }

    function ptMaterial(){
        return new THREE.ShaderMaterial({ uniforms:{uTex:{value:tex}},
            vertexShader:'attribute float size;\nattribute vec3 color;\nvarying vec3 vC;\nvoid main(){vC=color;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=size*(220.0/-mv.z);gl_Position=projectionMatrix*mv;}',
            fragmentShader:'uniform sampler2D uTex;\nvarying vec3 vC;\nvoid main(){vec4 t=texture2D(uTex,gl_PointCoord);if(t.a<0.02)discard;gl_FragColor=vec4(vC,1.0)*t;}',
            transparent:true,depthWrite:false,blending:THREE.AdditiveBlending });
    }
    async function показать(url){
        while(группа.children.length){ const o=группа.children.pop(); o.geometry?.dispose?.(); o.material?.dispose?.(); }
        segs=null; pts=null; wind=null;
        const r=await fetch(encodeURI(url),{cache:'no-store'}); if(!r.ok) throw new Error('модель HTTP '+r.status);
        model=await r.json();
        if(model.тип==='огонь' && model.parts){
            const P=model.parts, v=new Float32Array(P.length*3), c=new Float32Array(P.length*3), sz=new Float32Array(P.length);
            const g=new THREE.BufferGeometry();
            g.setAttribute('position',new THREE.BufferAttribute(v,3));
            g.setAttribute('color',new THREE.BufferAttribute(c,3));
            g.setAttribute('size',new THREE.BufferAttribute(sz,1));
            pts=new THREE.Points(g,ptMaterial()); группа.add(pts);
            камера.position.set(0,0.6,7);
        } else if(model.тип==='ветер' && model.parts){   // ВЕТЕР — горизонтальные струи (холодный поток)
            const W=model.parts, v=new Float32Array(W.length*6), c=new Float32Array(W.length*6);
            for(let i=0;i<W.length;i++){ const b=W[i].bright||0.8, col=[0.62*b,0.74*b,0.85*b];
                for(let q=0;q<2;q++){c[i*6+q*3]=col[0];c[i*6+q*3+1]=col[1];c[i*6+q*3+2]=col[2];} }
            const g=new THREE.BufferGeometry();
            g.setAttribute('position',new THREE.BufferAttribute(v,3)); g.setAttribute('color',new THREE.BufferAttribute(c,3));
            wind=new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.7,blending:THREE.AdditiveBlending,depthWrite:false}));
            группа.add(wind); камера.position.set(0,0.3,7);
        } else if((model.тип==='пульс'||model.тип==='дыхание') && model.частицы){   // СЕРДЦЕБИЕНИЕ / ДЫХАНИЕ — облако, расширяется от звука
            const P=model.частицы, pv=new Float32Array(P.length*3), pc=new Float32Array(P.length*3), ps=new Float32Array(P.length);
            const пульс=model.тип==='пульс';
            for(let i=0;i<P.length;i++){ const b=P[i].bright||0.8;
                if(пульс){ pc[i*3]=0.95*b; pc[i*3+1]=0.18*b; pc[i*3+2]=0.22*b; }   // сердце — красное
                else     { pc[i*3]=0.65*b; pc[i*3+1]=0.78*b; pc[i*3+2]=0.92*b; }   // дыхание — бело-голубое
                ps[i]=0.07+b*0.07; pv[i*3]=P[i].x; pv[i*3+1]=P[i].y; pv[i*3+2]=P[i].z; }
            const pg=new THREE.BufferGeometry(); pg.setAttribute('position',new THREE.BufferAttribute(pv,3));
            pg.setAttribute('color',new THREE.BufferAttribute(pc,3)); pg.setAttribute('size',new THREE.BufferAttribute(ps,1));
            pts=new THREE.Points(pg,ptMaterial()); группа.add(pts); камера.position.set(0,0,7);
        } else if(model.тип==='удар' && model.частицы){   // УДАР — рассыпанные фрагменты, вспышка+подскок на ударе
            const P=model.частицы, pv=new Float32Array(P.length*3), pc=new Float32Array(P.length*3), ps=new Float32Array(P.length);
            for(let i=0;i<P.length;i++){ const b=P[i].bright||0.7; pc[i*3]=0.62*b; pc[i*3+1]=0.54*b; pc[i*3+2]=0.42*b; ps[i]=0.06+b*0.07;
                pv[i*3]=P[i].x; pv[i*3+1]=P[i].y; pv[i*3+2]=P[i].z; }
            const pg=new THREE.BufferGeometry(); pg.setAttribute('position',new THREE.BufferAttribute(pv,3));
            pg.setAttribute('color',new THREE.BufferAttribute(pc,3)); pg.setAttribute('size',new THREE.BufferAttribute(ps,1));
            pts=new THREE.Points(pg,ptMaterial()); группа.add(pts); камера.position.set(0,0.6,7);
        } else if(model.тип==='ручей' && model.parts){   // РУЧЕЙ — горизонтальный поток воды + пузырьки
            const W=model.parts, v=new Float32Array(W.length*6), c=new Float32Array(W.length*6);
            for(let i=0;i<W.length;i++){ const b=W[i].bright||0.8, col=[0.35*b,0.70*b,0.88*b];
                for(let q=0;q<2;q++){c[i*6+q*3]=col[0];c[i*6+q*3+1]=col[1];c[i*6+q*3+2]=col[2];} }
            const g=new THREE.BufferGeometry();
            g.setAttribute('position',new THREE.BufferAttribute(v,3)); g.setAttribute('color',new THREE.BufferAttribute(c,3));
            wind=new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.8,blending:THREE.AdditiveBlending,depthWrite:false}));
            группа.add(wind);
            if(model.bubbles){ const M=model.bubbles, mv=new Float32Array(M.length*3), mc=new Float32Array(M.length*3), ms=new Float32Array(M.length);
                for(let i=0;i<M.length;i++){ const b=M[i].bright||0.7; mc[i*3]=0.7*b; mc[i*3+1]=0.9*b; mc[i*3+2]=1.0*b; ms[i]=0.05+b*0.06; }
                const mg=new THREE.BufferGeometry(); mg.setAttribute('position',new THREE.BufferAttribute(mv,3));
                mg.setAttribute('color',new THREE.BufferAttribute(mc,3)); mg.setAttribute('size',new THREE.BufferAttribute(ms,1));
                pts=new THREE.Points(mg,ptMaterial()); группа.add(pts);
            }
            камера.position.set(0,0.5,7);
        } else if(model.тип==='волны' && model.точки){   // ВОЛНЫ — колышущаяся поверхность с пеной
            const P=model.точки, pv=new Float32Array(P.length*3), pc=new Float32Array(P.length*3), ps=new Float32Array(P.length);
            for(let i=0;i<P.length;i++){ pc[i*3]=0.30; pc[i*3+1]=0.55; pc[i*3+2]=0.85; ps[i]=0.07; pv[i*3]=P[i].x; pv[i*3+1]=model.y_base||0; pv[i*3+2]=P[i].z; }
            const pg=new THREE.BufferGeometry(); pg.setAttribute('position',new THREE.BufferAttribute(pv,3));
            pg.setAttribute('color',new THREE.BufferAttribute(pc,3)); pg.setAttribute('size',new THREE.BufferAttribute(ps,1));
            pts=new THREE.Points(pg,ptMaterial()); группа.add(pts); камера.position.set(0,1.4,7);
        } else if(model.тип==='гром'){     // ГРОМ — молния (зигзаг) + грозовое облако
            if(model.болт && model.болт.length>1){ const B=model.болт, sN=B.length-1, v=new Float32Array(sN*6), c=new Float32Array(sN*6);
                for(let i=0;i<sN;i++){ v[i*6]=B[i].x; v[i*6+1]=B[i].y; v[i*6+2]=B[i].z||0; v[i*6+3]=B[i+1].x; v[i*6+4]=B[i+1].y; v[i*6+5]=B[i+1].z||0;
                    for(let q=0;q<2;q++){ c[i*6+q*3]=0.85; c[i*6+q*3+1]=0.90; c[i*6+q*3+2]=1.0; } }
                const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.BufferAttribute(v,3)); g.setAttribute('color',new THREE.BufferAttribute(c,3));
                segs=new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:1.0,blending:THREE.AdditiveBlending,depthWrite:false}));
                группа.add(segs);
            }
            if(model.частицы){ const P=model.частицы, pv=new Float32Array(P.length*3), pc=new Float32Array(P.length*3), ps=new Float32Array(P.length);
                for(let i=0;i<P.length;i++){ const b=P[i].bright||0.6; pc[i*3]=0.55*b; pc[i*3+1]=0.60*b; pc[i*3+2]=0.85*b; ps[i]=0.06+b*0.08;
                    pv[i*3]=P[i].x; pv[i*3+1]=P[i].y; pv[i*3+2]=P[i].z; }
                const pg=new THREE.BufferGeometry(); pg.setAttribute('position',new THREE.BufferAttribute(pv,3));
                pg.setAttribute('color',new THREE.BufferAttribute(pc,3)); pg.setAttribute('size',new THREE.BufferAttribute(ps,1));
                pts=new THREE.Points(pg,ptMaterial()); группа.add(pts);
            }
            камера.position.set(0,0,8);
        } else if(model.тип==='песок' && model.drops){   // ПЕСОК — сыпучая струя + горка
            const D=model.drops, v=new Float32Array(D.length*6), c=new Float32Array(D.length*6);
            for(let i=0;i<D.length;i++){ const b=D[i].bright||0.9, col=[0.86*b,0.72*b,0.46*b];
                for(let q=0;q<2;q++){c[i*6+q*3]=col[0];c[i*6+q*3+1]=col[1];c[i*6+q*3+2]=col[2];} }
            const g=new THREE.BufferGeometry();
            g.setAttribute('position',new THREE.BufferAttribute(v,3)); g.setAttribute('color',new THREE.BufferAttribute(c,3));
            segs=new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.9,blending:THREE.AdditiveBlending,depthWrite:false}));
            группа.add(segs);
            if(model.pile){ const M=model.pile, mv=new Float32Array(M.length*3), mc=new Float32Array(M.length*3), ms=new Float32Array(M.length);
                for(let i=0;i<M.length;i++){ const b=M[i].bright||0.8; mc[i*3]=0.86*b; mc[i*3+1]=0.72*b; mc[i*3+2]=0.46*b; ms[i]=0.06+b*0.06; mv[i*3]=M[i].x; mv[i*3+1]=M[i].y; mv[i*3+2]=M[i].z; }
                const mg=new THREE.BufferGeometry(); mg.setAttribute('position',new THREE.BufferAttribute(mv,3));
                mg.setAttribute('color',new THREE.BufferAttribute(mc,3)); mg.setAttribute('size',new THREE.BufferAttribute(ms,1));
                pts=new THREE.Points(mg,ptMaterial()); группа.add(pts);
            }
            камера.position.set(0,0,7);
        } else if(model.тип==='водопад' && model.drops){   // ВОДОПАД — плотный занавес + туман снизу
            const D=model.drops, v=new Float32Array(D.length*6), c=new Float32Array(D.length*6);
            for(let i=0;i<D.length;i++){ const b=D[i].bright||0.9, col=[0.62*b,0.82*b,1.0*b];
                for(let q=0;q<2;q++){c[i*6+q*3]=col[0];c[i*6+q*3+1]=col[1];c[i*6+q*3+2]=col[2];} }
            const g=new THREE.BufferGeometry();
            g.setAttribute('position',new THREE.BufferAttribute(v,3)); g.setAttribute('color',new THREE.BufferAttribute(c,3));
            segs=new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.85,blending:THREE.AdditiveBlending,depthWrite:false}));
            группа.add(segs);
            if(model.mist){ const M=model.mist, mv=new Float32Array(M.length*3), mc=new Float32Array(M.length*3), ms=new Float32Array(M.length);
                for(let i=0;i<M.length;i++){ const b=M[i].bright||0.7; mc[i*3]=0.70*b; mc[i*3+1]=0.85*b; mc[i*3+2]=1.0*b; ms[i]=0.06+b*0.07; }
                const mg=new THREE.BufferGeometry(); mg.setAttribute('position',new THREE.BufferAttribute(mv,3));
                mg.setAttribute('color',new THREE.BufferAttribute(mc,3)); mg.setAttribute('size',new THREE.BufferAttribute(ms,1));
                pts=new THREE.Points(mg,ptMaterial()); группа.add(pts);
            }
            камера.position.set(0,0.2,7);
        } else if(model.drops){     // дождь
            const D=model.drops, v=new Float32Array(D.length*6), c=new Float32Array(D.length*6);
            for(let i=0;i<D.length;i++){ const b=D[i].bright||0.8, col=[0.47*b,0.67*b,1.0*b];
                for(let q=0;q<2;q++){c[i*6+q*3]=col[0];c[i*6+q*3+1]=col[1];c[i*6+q*3+2]=col[2];} }
            const g=new THREE.BufferGeometry();
            g.setAttribute('position',new THREE.BufferAttribute(v,3)); g.setAttribute('color',new THREE.BufferAttribute(c,3));
            segs=new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.9,blending:THREE.AdditiveBlending,depthWrite:false}));
            группа.add(segs); камера.position.set(0,0.4,7);
        }
        controls.target.set(0,0,0); controls.update(); t0=performance.now();
    }
    return {
        узел,
        активировать(){ узел.classList.add('активна'); активна=true; resize(); raf=requestAnimationFrame(tick); },
        деактивировать(){ узел.classList.remove('активна'); активна=false; cancelAnimationFrame(raf); },
        показать,
        снять(){ ro.disconnect(); cancelAnimationFrame(raf); controls.dispose(); renderer.dispose(); узел.remove(); },
    };
}

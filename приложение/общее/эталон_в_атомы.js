// Тринити Альфа — ЭТАЛОН → АТОМЫ (слой V2).
// Реальный кадр-эталон (фото/видео-кадр) → сэмпл пикселей в атомы:
//   позиция и цвет — из картинки (атом ОПИСЫВАЕТ образ),
//   частота — из цвета по алфавиту (цвет↔Hz), громкость — из яркости.
// Атомы несут И образ, И звук. «Услышать образ» их сонифицирует.

const MAX_ATOMS = 4000;

function hue_to_freq(h){ return 20 * Math.pow(1000, (270 - h) / 270); }
function rgb2hsl(r,g,b){
    r/=255; g/=255; b/=255;
    const mx=Math.max(r,g,b), mn=Math.min(r,g,b), l=(mx+mn)/2, d=mx-mn;
    if(d===0) return [0,0,l];
    const s=d/(1-Math.abs(2*l-1));
    let h; if(mx===r) h=((g-b)/d)%6; else if(mx===g) h=(b-r)/d+2; else h=(r-g)/d+4;
    return [h*60, s, l];
}

// Картинка (HTMLImageElement) → атомы в схеме рендера (+ pos из пикселей)
export function эталон_в_атомы(img, опции={}){
    const SW=220, SH=Math.max(1, Math.round(SW * img.height / img.width));
    const cv=document.createElement('canvas'); cv.width=SW; cv.height=SH;
    const ctx=cv.getContext('2d'); ctx.drawImage(img,0,0,SW,SH);
    const px=ctx.getImageData(0,0,SW,SH).data;
    const порог=опции.порог ?? 36;     // отсечка фона (тёмное = пусто)
    const dur=опции.длительность ?? 2.0;
    const aspect=SW/SH;

    // собрать яркие пиксели
    const cand=[];
    for(let y=0;y<SH;y++) for(let x=0;x<SW;x++){
        const i=(y*SW+x)*4, r=px[i],g=px[i+1],b=px[i+2];
        if(Math.max(r,g,b)>порог) cand.push([x,y,r,g,b]);
    }
    // прорежаем до MAX_ATOMS
    let step=Math.max(1, Math.floor(cand.length/MAX_ATOMS));
    const atoms=[];
    for(let k=0;k<cand.length;k+=step){
        const [x,y,r,g,b]=cand[k];
        const [h,s,l]=rgb2hsl(r,g,b);
        const freq=hue_to_freq(h);                 // АЛФАВИТ: цвет → частота
        const ampN=Math.min(1, l*1.2);
        const birth=+( (y/SH)*dur ).toFixed(3);     // верх раньше (падение/разворот во времени)
        atoms.push({
            birth, freq:+freq.toFixed(1), amp:+ampN.toFixed(4),
            harmonic_index:0, harmonicity:0.08,      // эталон-картинка → шумовая фаза
            color_r:r, color_g:g, color_b:b,
            pos_x:+(((x/SW)-0.5)*2*aspect).toFixed(4),
            pos_y:+((0.5-(y/SH))*2).toFixed(4),
            pos_z:+(((l-0.5)*0.6)).toFixed(4),       // лёгкая глубина по яркости
        });
    }
    return { version:'etalon-1', parent_path: опции.имя||'эталон',
             длительность_сек:dur, atoms_count:atoms.length, atoms };
}

// ── поток: выбрать картинку → атомы → виртуальная клетка → «Образ» ──
window.__пользовательские = window.__пользовательские || { атомы:new Map(), звуки:new Map() };
let счёт=0, файл_инпут=null;
function загрузить_картинку(){
    if(!файл_инпут){
        файл_инпут=document.createElement('input');
        файл_инпут.type='file'; файл_инпут.accept='image/*'; файл_инпут.style.display='none';
        document.body.appendChild(файл_инпут);
        файл_инпут.addEventListener('change',()=>{
            const f=файл_инпут.files?.[0]; файл_инпут.value='';
            if(!f) return;
            const img=new Image();
            img.onload=()=>{ try{ показать(img, f.name.replace(/\.[^.]+$/,'')); }catch(e){ alert('Не удалось разобрать картинку: '+e.message); } };
            img.onerror=()=>alert('Не удалось загрузить изображение.');
            img.src=URL.createObjectURL(f);
        });
    }
    файл_инпут.click();
}
function показать(img, имя){
    const данные=эталон_в_атомы(img,{имя});
    if(!данные.atoms_count){ alert('Картинка тёмная — атомов не нашлось.'); return; }
    const id='etalon_'+(++счёт);
    window.__пользовательские.атомы.set(id, данные);
    const клеточка={ id, название:'Эталон: '+имя, группа:'моё', звук:'', сцена:'', атомы:'', решётка:'',
        число_атомов:данные.atoms_count, число_связей:0, длительность_сек:данные.длительность_сек,
        основной_элемент:'эталон', __в_памяти:true };
    window.__текущая_клеточка={ id, данные:клеточка };
    const b=document.querySelector('.прж[data-проекция="образ"]'); if(b) b.click();
    window.dispatchEvent(new CustomEvent('cellSelected',{ detail:{ id, данные:клеточка } }));
    const пасп=document.getElementById('паспорт');
    if(пасп) пасп.textContent=`${клеточка['название']} · ${данные.atoms_count.toLocaleString('ru-RU')} атомов · эталон`;
}

function подключить(){
    document.querySelectorAll('#панель-функций li').forEach((li)=>{
        if(li.dataset.id!=='etalon') return;
        li.addEventListener('click',(e)=>{ e.stopImmediatePropagation(); загрузить_картинку(); }, true);
    });
    window.__эталон_в_атомы=эталон_в_атомы;
}
if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',подключить);
else подключить();

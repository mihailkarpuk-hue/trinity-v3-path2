// Тринити V3 — ОБРАЗ → ЗВУК с сохранением вложенности.
// time-crosses → фазовая непрерывность; mod_depth/mod_rate → общая AM.

const SR = 22050;
const GRAIN = 0.06;
const HARM_THRESH = 0.30;
const MAX_ATOMS = 6000;

const SPARSE_MIN = 3000;
const SPARSE_DT = 0.06;

function atomScore(a) {
    return (a.amp || 0) * (0.3 + (a.harmonicity ?? 0));
}

function prepareAtoms(atoms, crosses, maxAtoms) {
    let list = atoms;
    let cr = crosses;
    if (list.length > SPARSE_MIN) {
        const dt = list.length > 4000 ? 0.08 : SPARSE_DT;
        const bins = new Map();
        list.forEach((a, i) => {
            const key = `${Math.round((a.birth || 0) / dt)}:${Math.round((a.freq || 0) / 15)}`;
            const cur = bins.get(key);
            if (!cur || atomScore(a) > atomScore(list[cur])) bins.set(key, i);
        });
        const keep = [...bins.values()].sort((a, b) => a - b);
        const map = new Map(keep.map((old, ni) => [old, ni]));
        list = keep.map(i => atoms[i]);
        if (cr?.length) {
            cr = cr.filter(c => map.has(c.atom_a) && map.has(c.atom_b))
                .map(c => ({ ...c, atom_a: map.get(c.atom_a), atom_b: map.get(c.atom_b) }));
        }
    }
    if (list.length > maxAtoms) {
        const keep = Array.from({ length: list.length }, (_, i) => i)
            .sort((a, b) => atomScore(list[b]) - atomScore(list[a]))
            .slice(0, maxAtoms)
            .sort((a, b) => a - b);
        const map = new Map(keep.map((old, ni) => [old, ni]));
        list = keep.map(i => list[i]);
        if (cr?.length) {
            cr = cr.filter(c => map.has(c.atom_a) && map.has(c.atom_b))
                .map(c => ({ ...c, atom_a: map.get(c.atom_a), atom_b: map.get(c.atom_b) }));
        }
    }
    return { list, crosses: cr };
}

function phasesFromCrosses(atoms, crosses) {
    const n = atoms.length;
    const phases = new Float64Array(n);
    const order = Array.from({ length: n }, (_, i) => i)
        .sort((a, b) => (atoms[a].birth ?? 0) - (atoms[b].birth ?? 0));
    for (let k = 1; k < order.length; k++) {
        const i0 = order[k - 1], i1 = order[k];
        const dt = (atoms[i1].birth ?? 0) - (atoms[i0].birth ?? 0);
        if (dt <= 0) continue;
        const favg = 0.5 * ((atoms[i0].freq ?? 440) + (atoms[i1].freq ?? 440));
        const h = Math.min(atoms[i0].harmonicity ?? 0, atoms[i1].harmonicity ?? 0);
        if (h > 0.45) phases[i1] = phases[i0] + 2 * Math.PI * favg * dt;
        else if (h > 0.15) phases[i1] = phases[i0] + 2 * Math.PI * favg * dt * 0.6 + (Math.random() - 0.5) * 0.8;
        else phases[i1] = phases[i0] * 0.35 + Math.random() * 2 * Math.PI * 0.65;
    }
    if (crosses?.length) {
        for (const c of crosses) {
            if (c.axis !== 'harmonic') continue;
            const ia = c.atom_a, ib = c.atom_b;
            if (ia >= n || ib >= n) continue;
            const fa = atoms[ia].freq ?? 0, fb = atoms[ib].freq ?? 0;
            if (fa > 0) phases[ib] = phases[ia] * (fb / fa);
        }
    }
    for (let i = 0; i < n; i++) {
        if ((atoms[i].harmonicity ?? 0) >= HARM_THRESH && atoms[i].phase != null)
            phases[i] = atoms[i].phase;
    }
    return phases;
}

function sharedEnv(N, modDepth, modRate) {
    const env = new Float32Array(N);
    if (modDepth <= 0.01 || modRate <= 0) { env.fill(1); return env; }
    for (let i = 0; i < N; i++) {
        const t = i / SR;
        env[i] = Math.max(0.05, 1 + modDepth * Math.sin(2 * Math.PI * modRate * t));
    }
    return env;
}

function синтез_буфер(данные) {
    const atoms = данные?.atoms || [];
    const crosses = данные?.crosses || null;
    const p104 = данные?.параметры104 || данные?.parent_params_full || {};
    const modDepth = +(p104.mod_depth ?? 0.3);
    const modRate = +(p104.mod_rate ?? 3.0);

    let dur = 0;
    for (const a of atoms) dur = Math.max(dur, (a.birth || 0));
    dur = Math.min(dur + GRAIN + 0.2, 30);
    const N = Math.ceil(dur * SR);
    const out = new Float32Array(N + SR);
    const shared = sharedEnv(N, modDepth, modRate);

    const { list, crosses: cr } = prepareAtoms(atoms, crosses, MAX_ATOMS);
    const phases = phasesFromCrosses(list, cr);
    const gl = Math.floor(GRAIN * SR);
    const env = new Float32Array(gl);
    for (let i = 0; i < gl; i++) env[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (gl - 1)));

    for (let ai = 0; ai < list.length; ai++) {
        const a = list[ai];
        const f = a.freq || 0;
        if (f <= 20 || f >= SR / 2) continue;
        const ampN = Math.min(1, (a.amp || 0) * 6);
        if (ampN <= 0) continue;
        const harm = a.harmonicity ?? 0.5;
        const ton = Math.max(0, Math.min(1, (harm - 0.05) / 0.45));
        const ph = ton > 0.2 ? phases[ai] : phases[ai] * ton + Math.random() * 2 * Math.PI * (1 - ton);
        const s = Math.floor((a.birth || 0) * SR);
        const w = 2 * Math.PI * f / SR;
        for (let i = 0; i < gl && s + i < N; i++)
            out[s + i] += ampN * 0.28 * env[i] * Math.sin(w * i + ph) * shared[s + i];
    }

    let m = 0;
    for (let i = 0; i < N; i++) { const v = Math.abs(out[i]); if (v > m) m = v; }
    if (m > 0) for (let i = 0; i < N; i++) out[i] = out[i] / m * 0.9;
    return { data: out.subarray(0, N), dur };
}

function floatToWav(data, sr) {
    const n = data.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const v = new DataView(buf);
    const ascii = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    ascii(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); ascii(8, 'WAVE'); ascii(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    ascii(36, 'data'); v.setUint32(40, n * 2, true);
    let o = 44;
    for (let i = 0; i < n; i++, o += 2) {
        const s = Math.max(-1, Math.min(1, data[i]));
        v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
}

export function wav_url_из_атомов(данные) {
    const { data } = синтез_буфер(данные);
    return URL.createObjectURL(floatToWav(data, SR));
}
window.__wav_url_из_атомов = wav_url_из_атомов;

let текущий_источник = null;
export async function сыграть_образ(данные) {
    const atoms = данные?.atoms || [];
    if (!atoms.length) { alert('В образе нет атомов для озвучивания.'); return; }
    const ctx = (window.__плеер && window.__плеер.audioCtx)
        ? window.__плеер.audioCtx
        : new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === 'suspended') { try { await ctx.resume(); } catch (e) {} }

    const { data, dur } = синтез_буфер(данные);
    const buf = ctx.createBuffer(1, data.length, SR);
    buf.getChannelData(0).set(data);

    if (текущий_источник) { try { текущий_источник.stop(); } catch (e) {} }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start();
    текущий_источник = src;
    console.log(`[образ→звук] атомов: ${atoms.length}, крестов: ${данные?.crosses?.length ?? 0}, ${dur.toFixed(1)}с`);
    return dur;
}

function подключить() {
    document.querySelectorAll('#панель-функций li').forEach((li) => {
        if (li.dataset.id !== 'hear') return;
        li.addEventListener('click', (e) => {
            e.stopImmediatePropagation();
            const данные = window.__текущие_атомы;
            if (!данные) { alert('Сначала выбери клеточку или запиши свой звук.'); return; }
            const кл = window.__текущая_клеточка?.данные;
            if (кл?.параметры104 && !данные.параметры104) данные.параметры104 = кл.параметры104;
            сыграть_образ(данные);
        }, true);
    });
    window.__образ_в_звук = сыграть_образ;
    window.__wav_url_из_атомов = wav_url_из_атомов;
}
if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', подключить);
else подключить();

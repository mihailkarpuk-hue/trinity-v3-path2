// Тринити V3 — озвучивание образа из словаря кирпичей (Этап 5).
// Только словарь + геометрия — без природных wav каталога.

import { атомыВОблако, облакоВСкелеты } from './образ_в_атомы.js';

const SR = 22050;
const FADE_MS = 2;
const ЖИВОСТЬ = 0.7;

const ТИП_КЛЮЧИ = [
    'фаза_тон', 'фаза_шум', 'фаза_переход', 'log_lifetime', 'freq_slope',
    'harmonicity', 'attack_ratio', 'decay_shape', 'noise_ratio',
    'band_width_rel', 'mod_depth', 'mod_rate',
];

let _словарь = null;
let _таблица = null;
let _источник = null;

async function загрузитьДанные() {
    if (_словарь && _таблица) return { словарь: _словарь, таблица: _таблица };
    const [s, t] = await Promise.all([
        fetch('../../данные/словарь_кирпичей.json').then(r => r.json()),
        fetch('../../данные/геометрия_в_кирпич.json').then(r => r.json()),
    ]);
    _словарь = s;
    _таблица = t;
    return { словарь: s, таблица: t };
}

async function sha256Bytes(buf) {
    const h = await crypto.subtle.digest('SHA-256', buf);
    return [...new Uint8Array(h)].map(b => b.toString(16).padStart(2, '0')).join('');
}

function seedAtom(birth, freq, typId) {
    const payload = `${birth}:${freq}:${typId}`;
    let h = 2166136261;
    for (let i = 0; i < payload.length; i++) {
        h ^= payload.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return h >>> 0;
}

function rng(seed) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(1664525, s) + 1013904223) >>> 0;
        return (s & 0xfffffff) / 0x10000000 - 1;
    };
}

function фазаИзRaw(v) {
    const i = v.slice(0, 3).reduce((bi, x, j) => x > v[bi] ? j : bi, 0);
    return ['тон', 'шум', 'переход'][i];
}

function genИзRaw(raw, birth, freq, amp) {
    const lt = Math.exp(raw[3]);
    const bw = Math.max(0.05, raw[9]) * freq;
    const h = Math.max(0, Math.min(1, raw[5]));
    return {
        фаза: фазаИзRaw(raw),
        birth, lifetime: lt, freq, freq_slope: raw[4],
        harmonic_index: h > 0.5 ? 1 : 0, harmonicity: h,
        amp, attack_ratio: Math.max(0, Math.min(1, raw[6])),
        decay_shape: raw[7], noise_ratio: Math.max(0, Math.min(1, raw[8])),
        band_center: freq, band_width: bw,
        mod_depth: Math.max(0, raw[10]), mod_rate: Math.max(0, raw[11]),
    };
}

function scoreТип(sk, typ, таб) {
    const пор = таб.пороги || {};
    const kLow = пор.кривизна_низкая ?? 0.15;
    const pHigh = пор.плотность_высокая ?? 2;
    const имя = typ.имя || '';
    let s = (typ.доля || 0) * 0.5;
    const крив = sk.лок_кривизна || 0;
    const плот = sk.лок_плотность || 1;
    const pools = таб.pools || {};
    if (крив <= kLow) {
        if (/тональная|чирп/.test(имя)) s += 3;
        if ((pools.тональные || []).includes(typ.id)) s += 1.5;
    } else {
        if (/шорох|шипящее|шумовое/.test(имя)) s += 3;
        if ((pools.шумовые || []).includes(typ.id)) s += 1.5;
    }
    if (плот >= pHigh) {
        if (/удар|пульс/.test(имя)) s += 2.5;
        if ((pools.ударные || []).includes(typ.id)) s += 1.5;
    }
    const freq = sk.freq || 440;
    const birth = sk.birth || 0;
    const targetH = Math.max(0.05, Math.min(0.95, 1 - Math.log10(Math.max(freq, 60)) / Math.log10(9000)));
    const th = typ.центроид?.harmonicity ?? 0.5;
    s += Math.max(0, 2.5 - Math.abs(th - targetH) * 5);
    s += ((birth * 1000 + freq + typ.id) % 1000) * 1e-4;
    return s;
}

function назначитьТип(sk, словарь, таб) {
    return (словарь.типы || []).reduce((best, t) =>
        scoreТип(sk, t, таб) > scoreТип(sk, best, таб) ? t : best);
}

function sampleGen(sk, typ, словарь, живость = ЖИВОСТЬ) {
    const rnd = rng(seedAtom(sk.birth, sk.freq, typ.id));
    const raw = ТИП_КЛЮЧИ.map((k, i) => {
        const c = typ.центроид[k];
        const sig = (typ.sigma[k] || 0) * живость;
        return c + rnd() * sig;
    });
    return genИзRaw(raw, sk.birth, sk.freq, sk.amp);
}

function hann(n) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
    return w;
}

function synthGen(g, sr) {
    const fade = Math.max(2, Math.floor(FADE_MS * 1e-3 * sr));
    const n = Math.max(Math.floor(g.lifetime * sr), fade * 2);
    const env = new Float64Array(n).fill(1);
    const fi = Math.max(1, Math.floor(g.attack_ratio * n));
    for (let i = 0; i < fi; i++) env[i] = i / fi;
    const tail = n - fi;
    if (tail > 0 && g.decay_shape < 0) {
        for (let i = 0; i < tail; i++) env[fi + i] = Math.exp(g.decay_shape * i / sr);
    }
    const hw = hann(fade * 2);
    for (let i = 0; i < fade; i++) { env[i] *= hw[i]; env[n - fade + i] *= hw[fade + i]; }
    const ton = Math.max(0, Math.min(1, g.harmonicity * (1 - g.noise_ratio)));
    const sig = new Float64Array(n);
    const w = 2 * Math.PI * g.freq / sr;
    let ph = 0;
    for (let i = 0; i < n; i++) {
        if (ton > 0.05) sig[i] += ton * Math.sin(ph) * env[i] * g.amp * 0.35;
        ph += w;
    }
    return { s0: Math.floor(g.birth * sr), sig };
}

function synthGens(gens, sr = SR) {
    const dur = gens.reduce((d, g) => Math.max(d, g.birth + g.lifetime), 0) + 0.05;
    const N = Math.ceil(dur * sr) + 2048;
    const out = new Float32Array(N);
    for (const g of gens) {
        const { s0, sig } = synthGen(g, sr);
        for (let i = 0; i < sig.length && s0 + i < N; i++) out[s0 + i] += sig[i];
    }
    let m = 0;
    for (let i = 0; i < N; i++) if (Math.abs(out[i]) > m) m = Math.abs(out[i]);
    if (m > 0) for (let i = 0; i < N; i++) out[i] = out[i] / m * 0.9;
    return { data: out, dur };
}

export async function собратьИзСкелетов(скелеты, { живость = ЖИВОСТЬ } = {}) {
    const { словарь, таблица } = await загрузитьДанные();
    const gens = [];
    const counts = {};
    for (const sk of скелеты) {
        const typ = назначитьТип(sk, словарь, таблица);
        gens.push(sampleGen(sk, typ, словарь, живость));
        counts[typ.имя] = (counts[typ.имя] || 0) + 1;
    }
    const total = скелеты.length || 1;
    const состав = Object.fromEntries(
        Object.entries(counts).map(([k, v]) => [k, Math.round(v / total * 1000) / 1000]));
    return { gens, состав, synth: synthGens(gens) };
}

export async function озвучитьСкелеты(скелеты, opts) {
    const { synth, состав } = await собратьИзСкелетов(скелеты, opts);
    const ctx = window.__плеер?.audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === 'suspended') await ctx.resume();
    const buf = ctx.createBuffer(1, synth.data.length, SR);
    buf.getChannelData(0).set(synth.data);
    if (_источник) { try { _источник.stop(); } catch (e) {} }
    _источник = ctx.createBufferSource();
    _источник.buffer = buf;
    _источник.connect(ctx.destination);
    _источник.start();
    показатьСостав(состав);
    return { dur: synth.dur, состав };
}

export async function озвучитьИзАтомов(данные, opts) {
    const atoms = данные?.atoms || [];
    if (!atoms.length) throw new Error('нет атомов');
    const { облако, meta } = атомыВОблако(atoms, { канон: false });
    const sk = облакоВСкелеты(облако, { meta, ограничитьПлотность: false });
    return озвучитьСкелеты(sk, opts);
}

function показатьСостав(состав) {
    let el = document.getElementById('плашка-состава');
    if (!el) {
        el = document.createElement('div');
        el.id = 'плашка-состава';
        el.style.cssText = 'position:fixed;bottom:72px;left:50%;transform:translateX(-50%);z-index:40;' +
            'background:#1a1a1aee;border:1px solid #444;color:#eee;padding:8px 14px;border-radius:8px;' +
            'font-size:12px;max-width:90vw';
        document.body.appendChild(el);
    }
    const parts = Object.entries(состав)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`)
        .join(' · ');
    el.textContent = `Кирпичи: ${parts || '—'}`;
    el.hidden = false;
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.hidden = true; }, 8000);
}

function подключить() {
    const btn = document.createElement('li');
    btn.dataset.id = 'brick_sound';
    btn.textContent = '🧱 Озвучить образ';
    const panel = document.getElementById('панель-функций');
    if (panel) panel.insertBefore(btn, panel.querySelector('[data-id="hear"]')?.nextSibling || null);

    btn?.addEventListener('click', async (e) => {
        e.stopImmediatePropagation();
        try {
            const данные = window.__текущие_атомы;
            if (!данные?.atoms?.length) {
                alert('Сначала выбери клеточку с атомами.');
                return;
            }
            await озвучитьИзАтомов(данные, { живость: ЖИВОСТЬ });
        } catch (err) {
            console.error(err);
            alert('Ошибка сборки: ' + err.message);
        }
    }, true);

    window.__озвучить_образ = { озвучитьИзАтомов, озвучитьСкелеты, собратьИзСкелетов };
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', подключить);
else подключить();

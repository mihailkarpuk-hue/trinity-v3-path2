// Тринити V3 — обратная проекция: облако образа → скелеты атомов.
// Зеркало ядро/образ_в_атомы.py (единая логика с offline-тестами).

const X_SCALE = 1.8;
const Y_SCALE = 2.4;
const Z_SCALE = 1.2;
const T_ОБРАЗА = 3.0;
const F_LO = 60;
const F_HI = 9000;
const AMP_MIN = 0.05;
const AMP_MAX = 1.0;
const ОКНО_BIRTH = 0.05;
const MAX_СКЕЛЕТОВ = 24;
const K = 8;

function clip01(v) { return Math.max(0, Math.min(1, v)); }
function norm(v, a, b) { return b === a ? 0.5 : (v - a) / (b - a); }

export function границыАтомов(atoms) {
    const births = atoms.map(a => a.birth || 0);
    const lfs = atoms.map(a => Math.log10(Math.max(1, a.freq || 20)));
    const amps = atoms.map(a => a.amp || 0);
    return {
        tmin: Math.min(...births), tmax: Math.max(...births),
        fmin: Math.min(...lfs), fmax: Math.max(...lfs),
        amax: Math.max(...amps, 1e-6),
    };
}

export function атомыВОблако(atoms, { канон = false, tОбраза = T_ОБРАЗА } = {}) {
    if (!atoms?.length) return { облако: [], meta: {} };
    const b = границыАтомов(atoms);
    const meta = канон
        ? { режим: 'канон', t_образа: tОбраза }
        : { ...b, режим: 'физика', t_образа: tОбраза };
    const облако = atoms.map(a => {
        const birth = a.birth || 0;
        const freq = Math.max(1, a.freq || 20);
        const amp = a.amp || 0;
        let x, y, z;
        if (канон) {
            const nx = clip01(birth / tОбраза);
            x = (2 * nx - 1) * X_SCALE;
            const ny = clip01((Math.log(freq) - Math.log(F_LO)) / (Math.log(F_HI) - Math.log(F_LO)));
            y = (ny - 0.5) * Y_SCALE;
            z = clip01((amp - AMP_MIN) / (AMP_MAX - AMP_MIN)) * Z_SCALE;
        } else {
            x = (2 * norm(birth, b.tmin, b.tmax) - 1) * X_SCALE;
            y = (norm(Math.log10(freq), b.fmin, b.fmax) - 0.5) * Y_SCALE;
            z = (amp / b.amax) * Z_SCALE;
        }
        const pt = { x, y, z, size: a.size ?? Math.max(0.05, amp) };
        if (a.color_r != null) { pt.r = a.color_r; pt.g = a.color_g; pt.b = a.color_b; }
        return pt;
    });
    return { облако, meta };
}

function точкаВСкелет(pt, meta) {
    const { x, y, z } = pt;
    const w = pt.size ?? 1;
    let birth, freq, amp;
    if (meta.режим === 'физика' && meta.tmin != null) {
        birth = ((x / X_SCALE + 1) / 2) * (meta.tmax - meta.tmin) + meta.tmin;
        const logf = (y / Y_SCALE + 0.5) * (meta.fmax - meta.fmin) + meta.fmin;
        freq = 10 ** logf;
        amp = (z / Z_SCALE) * meta.amax;
    } else {
        const t = meta.t_образа ?? T_ОБРАЗА;
        birth = clip01((x / X_SCALE + 1) / 2) * t;
        const ny = clip01(y / Y_SCALE + 0.5);
        freq = Math.exp(ny * (Math.log(F_HI) - Math.log(F_LO)) + Math.log(F_LO));
        amp = AMP_MIN + clip01(z / Z_SCALE) * (AMP_MAX - AMP_MIN);
    }
    const sk = {
        birth: Math.max(0, birth),
        freq: Math.max(F_LO, Math.min(F_HI, freq)),
        amp: meta.режим === 'физика' ? Math.max(0, amp) : Math.max(AMP_MIN, Math.min(AMP_MAX, amp)),
        _w: w,
    };
    if (pt.r != null) sk.цвет = { r: pt.r, g: pt.g, b: pt.b };
    return sk;
}

function слитьОкно(group) {
    const wsum = group.reduce((s, g) => s + g._w, 0) || 1;
    return {
        birth: group.reduce((s, g) => s + g.birth * g._w, 0) / wsum,
        freq: group.reduce((s, g) => s + g.freq * g._w, 0) / wsum,
        amp: group.reduce((s, g) => s + g.amp * g._w, 0) / wsum,
        _w: wsum,
        ...(group[0].цвет ? { цвет: group[0].цвет } : {}),
    };
}

function ограничитьПлотность(скелеты) {
    const sorted = [...скелеты].sort((a, b) => a.birth - b.birth);
    const out = [];
    let i = 0;
    while (i < sorted.length) {
        const t0 = sorted[i].birth;
        const group = [];
        while (i < sorted.length && sorted[i].birth - t0 <= ОКНО_BIRTH + 1e-9) group.push(sorted[i++]);
        if (group.length <= MAX_СКЕЛЕТОВ) out.push(...group);
        else {
            group.sort((a, b) => b._w - a._w);
            out.push(...group.slice(0, MAX_СКЕЛЕТОВ));
            if (group.length > MAX_СКЕЛЕТОВ) out.push(слитьОкно(group.slice(MAX_СКЕЛЕТОВ)));
        }
    }
    return out;
}

function dist3(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function локКривизна(pts, i) {
    if (pts.length < 3) return 0;
    const d = pts.map((p, j) => ({ j, d: j === i ? Infinity : dist3(p, pts[i]) }));
    d.sort((a, b) => a.d - b.d);
    const nn = d.slice(0, Math.min(K, pts.length - 1)).map(x => pts[x.j]);
    if (nn.length < 2) return 0;
    const cx = nn.reduce((s, p) => s + p[0], 0) / nn.length;
    const cy = nn.reduce((s, p) => s + p[1], 0) / nn.length;
    const cz = nn.reduce((s, p) => s + p[2], 0) / nn.length;
    let sxx = 0, syy = 0, szz = 0;
    for (const p of nn) {
        sxx += (p[0] - cx) ** 2;
        syy += (p[1] - cy) ** 2;
        szz += (p[2] - cz) ** 2;
    }
    const trace = (sxx + syy + szz) / nn.length;
    return trace > 1e-12 ? Math.min(sxx, syy, szz) / (sxx + syy + szz) : 0;
}

function локПлотность(pts, i) {
    if (pts.length <= 1) return 1;
    const d = pts.map((p, j) => j === i ? Infinity : dist3(p, pts[i])).sort((a, b) => a - b);
    const r = d.slice(0, Math.min(K, pts.length - 1)).reduce((s, v) => s + v, 0) / Math.min(K, pts.length - 1);
    return 1 / Math.max(r, 1e-3);
}

export function облакоВСкелеты(облако, { meta = null, tОбраза = T_ОБРАЗА, ограничитьПлотность = true } = {}) {
    if (!облако?.length) return [];
    const m = meta ? { ...meta } : { режим: 'канон', t_образа: tОбраза };
    if (!m.режим) m.режим = 'канон';
    let raw = облако.map(p => точкаВСкелет(p, m));
    if (ограничитьПлотность) raw = ограничитьПлотность(raw);
    const pts = raw.map(s => [s.birth, Math.log(Math.max(s.freq, 1)), s.amp]);
    return raw.map((s, i) => ({
        birth: s.birth,
        freq: s.freq,
        amp: s.amp,
        лок_кривизна: локКривизна(pts, i),
        лок_плотность: локПлотность(pts, i),
        ...(s.цвет ? { цвет: s.цвет } : {}),
    }));
}

export function образВАтомы(облако, opts) {
    return облакоВСкелеты(облако, opts);
}

/** Пиксели канваса → точки (y инвертирован, z = яркость). */
export function изКанваса(imageData, { порог = 30, шаг = 2 } = {}) {
    const { data, width, height } = imageData;
    const pts = [];
    for (let y = 0; y < height; y += шаг) {
        for (let x = 0; x < width; x += шаг) {
            const i = (y * width + x) * 4;
            const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
            const ярк = (r + g + b) / 3;
            if (a < 10 || ярк < порог) continue;
            pts.push({
                x: (x / width) * 2 * X_SCALE - X_SCALE,
                y: ((height - y) / height - 0.5) * Y_SCALE * 2,
                z: (ярк / 255) * Z_SCALE,
                size: ярк / 255,
                r, g, b,
            });
        }
    }
    return pts;
}

if (typeof window !== 'undefined') {
    window.__образ_в_атомы = { атомыВОблако, облакоВСкелеты, образВАтомы, изКанваса };
}

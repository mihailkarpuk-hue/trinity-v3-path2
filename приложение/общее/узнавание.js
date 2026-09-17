// Тринити V3 — УЗНАВАНИЕ (слой 2: «услышал → представил ближайший образ»).
// Метрика d₉ (блочная по 9 группам) — зеркало ядро/метрика.py.

let ИНДЕКС = null;
fetch(encodeURI('../../данные/узнавание_индекс.json'))
    .then(r => r.json()).then(d => { ИНДЕКС = d; })
    .catch(e => console.warn('[узнавание] индекс не загружен', e));

let ИНДЕКС108 = null;
let ВЕСА_ГРУПП = null;
let КЛЮЧИ_КАНОН = null;
let _группыИндексы = null;

const ГРУППЫ_ПОРЯДОК = [
    'TEMPORAL', 'SPECTRAL', 'VOCAL', 'MUSICAL', 'SPATIAL',
    'PERCEPTUAL', 'MOVEMENT', 'ADVANCED', 'AXES',
];

Promise.all([
    fetch(encodeURI('../../данные/узнавание_индекс_108.json')).then(r => r.json()),
    fetch(encodeURI('../../данные/веса_групп.json')).then(r => r.json()),
    fetch(encodeURI('../../данные/ключи_108.json')).then(r => r.json()),
]).then(([idx, веса, ключи]) => {
    if (idx.версия >= 2 && ключи.sha256 && idx.ключи_sha256 !== ключи.sha256) {
        console.error('[узнавание] расхождение ключи_sha256: индекс vs канон — откат на 5 фич');
        ИНДЕКС108 = null;
        return;
    }
    ИНДЕКС108 = idx;
    ВЕСА_ГРУПП = веса;
    КЛЮЧИ_КАНОН = ключи;
    _построитьИндексыГрупп();
}).catch(e => console.warn('[узнавание] индекс-108/веса не загружены', e));

function _построитьИндексыГрупп() {
    if (!ИНДЕКС108 || !КЛЮЧИ_КАНОН) return;
    const pos = Object.fromEntries(ИНДЕКС108.оси.map((k, i) => [k, i]));
    _группыИндексы = {};
    for (const g of ГРУППЫ_ПОРЯДОК) {
        const keys = КЛЮЧИ_КАНОН.группы[g] || [];
        _группыИндексы[g] = keys.map(k => pos[k]).filter(i => i != null);
    }
}

function _dG(a, b, inds, mu, sd) {
    let s = 0;
    for (const i of inds) {
        const za = (a[i] - mu[i]) / sd[i];
        const zb = (b[i] - mu[i]) / sd[i];
        const d = za - zb;
        s += d * d;
    }
    return Math.sqrt(s);
}

function d9(a, b, mu, sd) {
    let total = 0;
    for (const g of ГРУППЫ_ПОРЯДОК) {
        const inds = _группыИндексы[g];
        if (!inds || !inds.length) continue;
        const dg = _dG(a, b, inds, mu, sd);
        const w = (ВЕСА_ГРУПП && ВЕСА_ГРУПП[g] != null) ? +ВЕСА_ГРУПП[g] : 1;
        total += w * dg / Math.sqrt(inds.length);
    }
    return total;
}

function вектор108(atoms, оси) {
    const с_паспортом = atoms.filter(a => a && a.params_104 && Object.keys(a.params_104).length >= 100);
    if (с_паспортом.length < 1) return null;
    const v = new Array(оси.length).fill(0);
    for (const a of с_паспортом) {
        const p = a.params_104;
        for (let i = 0; i < оси.length; i++) {
            const x = +p[оси[i]];
            if (Number.isFinite(x)) v[i] += x;
        }
    }
    const n = с_паспортом.length;
    return v.map(x => x / n);
}

function ближайшие108(atoms, k = 4) {
    if (!ИНДЕКС108 || !_группыИндексы) return null;
    const q = вектор108(atoms, ИНДЕКС108.оси);
    if (!q) return null;
    const mu = ИНДЕКС108.mu, sd = ИНДЕКС108.sd;
    const res = ИНДЕКС108.клетки.map(c => ({
        id: c.id,
        название: c.название,
        группа: c.группа,
        d: d9(q, c.вектор, mu, sd),
    }));
    res.sort((a, b) => a.d - b.d);
    return res.slice(0, k);
}

function фичи(atoms) {
    let ton = 0, noi = 0, tr = 0, slf = 0, sh = 0;
    const n = atoms.length || 1;
    for (const a of atoms) {
        const h = (a.harmonicity != null) ? a.harmonicity : 0.5;
        if (h > 0.6) ton++; else if (h < 0.05) noi++; else tr++;
        slf += Math.log10(Math.max(80, a.freq || 100));
        sh += h;
    }
    return [ton / n, noi / n, tr / n, slf / n, sh / n];
}

function ближайшие(atoms, k = 4) {
    if (!ИНДЕКС) return null;
    const q = фичи(atoms);
    const mu = ИНДЕКС.mu, sd = ИНДЕКС.sd;
    const zq = q.map((v, i) => (v - mu[i]) / sd[i]);
    const res = ИНДЕКС.клетки.map(c => {
        let d = 0;
        for (let i = 0; i < 5; i++) { const zc = (c.вектор[i] - mu[i]) / sd[i]; d += (zq[i] - zc) ** 2; }
        return { id: c.id, название: c.название, группа: c.группа, d: Math.sqrt(d) };
    });
    res.sort((a, b) => a.d - b.d);
    return res.slice(0, k);
}

export function узнать(atoms) {
    if (!atoms || !atoms.length) { alert('Сначала выбери клеточку, запиши или загрузи звук.'); return; }
    const top108 = ближайшие108(atoms, 4);
    const top = top108 || ближайшие(atoms, 4);
    if (!top) { alert('Индекс узнавания ещё загружается, попробуй через секунду.'); return; }
    const глубина = top108 ? 'd₉ · 108 параметров · блочная метрика' : 'фаза · яркость · тембр';
    const список = top.filter(x => x.d > 0.001).slice(0, 3);
    const строки = список.map((x, i) => `${i + 1}. ${x.название}  ·  ${x.группа}  (d=${x.d.toFixed(2)})`).join('\n');
    alert('🧠 Этот звук ближе всего к образам:\n\n' + строки + '\n\n(узнавание — ' + глубина + ' — слой 2)');
    return список;
}

function подключить() {
    document.querySelectorAll('#панель-функций li').forEach((li) => {
        if (li.dataset.id !== 'recognize') return;
        li.addEventListener('click', (e) => {
            e.stopImmediatePropagation();
            узнать(window.__текущие_атомы?.atoms || window.__текущие_атомы);
        }, true);
    });
    window.__узнать = узнать;
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', подключить);
else подключить();

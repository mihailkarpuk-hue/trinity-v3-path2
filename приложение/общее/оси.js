// оси.js — JS-порт фрактальных осей (шов 2: один язык формул в Python и JS).
// Формулы 1:1 с ядро/оси.py. Чтобы загруженный пользователем звук в браузере
// получал те же оси, что и корпус (один источник истины по формулам).
//
// ПОРТИРОВАНО ТОЧНО: higuchi (FD + R²), модуляция (глубина).
// НЕ портировано (тяжёлый DSP Гильберта/Баттерворта — остаётся Python):
//   вложенность (корреляция огибающих полос). Для загруженного звука
//   вложенность берётся либо с сервера, либо помечается как «—».

// ── FD Хигучи + самоподобие R² (точный порт higuchi из оси.py) ──
export function higuchi(x, kmax = 12) {
    const N = x.length;
    const pts = [];
    for (let k = 1; k <= kmax; k++) {
        const Lk = [];
        for (let m = 0; m < k; m++) {
            const idx = [];
            for (let i = m; i < N; i += k) idx.push(i);
            if (idx.length < 2) continue;
            let s = 0;
            for (let j = 1; j < idx.length; j++) s += Math.abs(x[idx[j]] - x[idx[j - 1]]);
            Lk.push(s * (N - 1) / ((idx.length - 1) * k));
        }
        if (Lk.length) {
            const mean = Lk.reduce((a, b) => a + b, 0) / Lk.length;
            pts.push([Math.log(1.0 / k), Math.log(mean)]);
        }
    }
    // линейная регрессия наклона + R²
    const n = pts.length;
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const [px, py] of pts) { sx += px; sy += py; sxx += px * px; sxy += px * py; }
    const a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const b = (sy - a * sx) / n;
    const ym = sy / n;
    let ssr = 0, sst = 0;
    for (const [px, py] of pts) { const f = a * px + b; ssr += (py - f) ** 2; sst += (py - ym) ** 2; }
    const r2 = sst > 0 ? 1 - ssr / sst : 0;
    return { fd: a, selfsim_r2: r2 };
}

// ── модуляция: глубина дыхания огибающей (простой порт; огибающая = |x| сглаж.)
export function модуляция(x, sr) {
    // огибающая через скользящее RMS-окно ~12.5 мс (аналог огибающей оси.py)
    const w = Math.max(8, Math.round(sr * 0.0125));
    const e = new Float32Array(x.length);
    let acc = 0;
    for (let i = 0; i < x.length; i++) {
        acc += x[i] * x[i];
        if (i >= w) acc -= x[i - w] * x[i - w];
        e[i] = Math.sqrt(acc / Math.min(i + 1, w));
    }
    let mean = 0; for (let i = 0; i < e.length; i++) mean += e[i]; mean /= e.length;
    let v = 0; for (let i = 0; i < e.length; i++) v += (e[i] - mean) ** 2; v /= e.length;
    return { mod_depth: Math.sqrt(v) / (mean + 1e-9) };
}

// ── единый вызов: оси из waveform (нормировать заранее) ──
export function оси_звука(x, sr) {
    let m = 0; for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
    if (m > 0) { x = Float32Array.from(x, v => v / m); }
    const h = higuchi(x);
    const md = модуляция(x, sr);
    return {
        fd: +h.fd.toFixed(3),
        selfsim_r2: +h.selfsim_r2.toFixed(3),
        mod_depth: +md.mod_depth.toFixed(3),
        nestedness: null,   // тяжёлый DSP — берётся с сервера (см. оси.py)
    };
}

/*
КАК ВСТАВИТЬ (шов 2) в приложение/общее/атомизация.js:
  import { оси_звука } from './оси.js';
  // после получения сэмплов x (Float32Array) и sr:
  const оси = оси_звука(x, sr);   // {fd, selfsim_r2, mod_depth, nestedness:null}
  данные.параметры104 = { ...(данные.параметры104||{}), ...оси };
Так загруженный звук получит FD/R²/модуляцию теми же формулами, что корпус.
Вложенность для загруженного — null (нужен серверный проход оси.py).
*/

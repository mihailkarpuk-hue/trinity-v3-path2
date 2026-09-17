// Тринити Альфа — Атомизатор «на лету».
// AudioBuffer → массив атомов В СХЕМЕ РЕНДЕРА (проекция_3d.собрать_контент):
//   { birth, freq, amp, harmonic_index, color_r, color_g, color_b,
//     harmonicity, phase }
// Самодостаточен: свой FFT, без зависимости от silhouettes.js.
// Так «свой голос» можно нарисовать ТЕМ ЖЕ движком, что и 218 клеточек.

const FFT_SIZE = 2048;
const HOP_MS   = 16;     // ~60 кадров/с
const TOP_PEAKS = 36;    // пиков (атомов) на кадр
const MIN_AMP  = 0.004;  // отсечка тихих пиков (после нормировки на N/2)

// ─── FFT (Cooley-Tukey, radix-2, in-place) ───────────────────────────────
function fft(re, im) {
    const N = re.length;
    for (let i = 1, j = 0; i < N; i++) {
        let bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let size = 2; size <= N; size <<= 1) {
        const half = size >> 1, step = 2 * Math.PI / size;
        for (let i = 0; i < N; i += size) {
            for (let j = i, k = 0; j < i + half; j++, k++) {
                const ang = -step * k, c = Math.cos(ang), s = Math.sin(ang);
                const tRe = re[j + half] * c - im[j + half] * s;
                const tIm = re[j + half] * s + im[j + half] * c;
                re[j + half] = re[j] - tRe; im[j + half] = im[j] - tIm;
                re[j] += tRe;               im[j] += tIm;
            }
        }
    }
}
function hann(buf) {
    const N = buf.length;
    for (let i = 0; i < N; i++) buf[i] *= 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
}

// ─── частота → цвет (как в материнском sound-atoms: фиолет низ → красный верх) ─
function freqToHue(hz) {
    const t = Math.log10(Math.max(20, Math.min(hz, 20000)) / 20) / Math.log10(1000);
    return 270 - Math.min(t, 1) * 270;
}
function hslToRgb(h, s, l) {
    h /= 360;
    const a = s * Math.min(l, 1 - l);
    const f = n => { const k = (n + h * 12) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
    return [f(0), f(8), f(4)];
}

// ─── Главная: AudioBuffer → атомы ────────────────────────────────────────
export function атомизировать(audioBuffer, опции = {}) {
    const data = audioBuffer.getChannelData(0);
    const sr = audioBuffer.sampleRate;
    const N = FFT_SIZE;
    const hop = Math.max(1, Math.floor(sr * HOP_MS / 1000));
    const totalFrames = Math.max(0, Math.floor((data.length - N) / hop));
    const maxAtoms = опции.maxAtoms || 12000;
    const красить_по_фазе = !!опции.красить_по_фазе;

    const атомы = [];
    const re = new Float32Array(N), im = new Float32Array(N);
    const nyquist = sr / 2, binToHz = nyquist / (N / 2);
    let prevEnergy = 0, baseline = 0;   // для детекции транзиентов (онсетов)

    for (let frame = 0; frame < totalFrames; frame++) {
        if (атомы.length >= maxAtoms) break;
        const off = frame * hop;
        for (let i = 0; i < N; i++) re[i] = data[off + i] || 0;
        im.fill(0);
        hann(re); fft(re, im);

        const tSec = frame * HOP_MS / 1000;

        // Пики-локальные максимумы + энергия кадра
        const peaks = [];
        let energy = 0;
        let prevMag = 0, curMag = Math.hypot(re[2], im[2]) / (N / 2);
        for (let bin = 2; bin < N / 2 - 1; bin++) {
            const nextMag = Math.hypot(re[bin + 1], im[bin + 1]) / (N / 2);
            energy += curMag;
            if (curMag >= MIN_AMP && curMag > prevMag && curMag > nextMag) {
                peaks.push({ freq: bin * binToHz, amp: curMag });
            }
            prevMag = curMag; curMag = nextMag;
        }

        // ── ТРАНЗИЕНТ: резкий скачок энергии → широкополосная вспышка (щелчок/удар) ──
        const jump = energy - prevEnergy;
        const isOnset = baseline > 0 && jump > 0.6 * baseline && energy > 1.5 * baseline;
        if (isOnset) {
            const M = 14, fLo = 40, fHi = Math.min(nyquist, 16000);
            for (let b = 0; b < M && атомы.length < maxAtoms; b++) {
                const lo = fLo * Math.pow(fHi / fLo, b / M), hi = fLo * Math.pow(fHi / fLo, (b + 1) / M);
                let e = 0;
                const bl = Math.max(2, Math.floor(lo / binToHz)), bh = Math.min(N / 2 - 1, Math.ceil(hi / binToHz));
                for (let bin = bl; bin < bh; bin++) e += Math.hypot(re[bin], im[bin]) / (N / 2);
                if (e <= 0) continue;
                const fc = Math.sqrt(lo * hi), amp = Math.min(1, e * 6);
                const [rc, gc, bc] = hslToRgb(freqToHue(fc), 0.12 + 0.05 * 0.85, 0.45 + amp * 0.40);
                атомы.push({ birth: tSec, freq: fc, amp, harmonic_index: 0, harmonicity: 0.05,
                    phase: 'шум', color_r: Math.round(rc * 255), color_g: Math.round(gc * 255), color_b: Math.round(bc * 255) });
            }
        }
        prevEnergy = energy;
        baseline = baseline === 0 ? energy : baseline * 0.95 + energy * 0.05;

        if (!peaks.length) continue;
        peaks.sort((a, b) => b.amp - a.amp);
        const top = peaks.slice(0, TOP_PEAKS);

        // Гармоничность кадра + основной тон f0 (самый низкий «сильный» пик)
        const sumAmp = top.reduce((s, p) => s + p.amp, 0) + 1e-9;
        const harmonicity = top[0].amp / sumAmp;            // 0..1: 1 = один чистый пик
        const порог_f0 = top[0].amp * 0.35;
        let f0 = Infinity;
        for (const p of top) if (p.amp >= порог_f0 && p.freq < f0) f0 = p.freq;
        if (!isFinite(f0) || f0 < 20) f0 = top[0].freq;

        const phase = harmonicity > 0.6 ? 'тон' : (harmonicity < 0.2 ? 'шум' : 'переход');

        for (const p of top) {
            if (атомы.length >= maxAtoms) break;
            const ampNorm = Math.min(1, p.amp * 6);
            const hi = f0 > 0 ? Math.max(0, Math.min(64, Math.round(p.freq / f0))) : 0;

            let r, g, b;
            if (красить_по_фазе) {
                // тон → золото, переход → оранжевый, шум → синий (грамматика фаз Гиббса)
                const c = phase === 'тон' ? [255, 215, 0] : phase === 'переход' ? [255, 99, 71] : [30, 144, 255];
                const k = 0.45 + ampNorm * 0.55;
                r = c[0] * k; g = c[1] * k; b = c[2] * k;
            } else {
                // насыщенность = ТЕМБР (буква алфавита): шум→серый, тон→насыщенный
                const [rc, gc, bc] = hslToRgb(freqToHue(p.freq), 0.12 + harmonicity * 0.85, 0.45 + ampNorm * 0.40);
                r = rc * 255; g = gc * 255; b = bc * 255;
            }
            атомы.push({
                birth: tSec,
                freq: p.freq,
                amp: ampNorm,
                harmonic_index: hi,
                harmonicity: +harmonicity.toFixed(3),
                phase,
                color_r: Math.round(r), color_g: Math.round(g), color_b: Math.round(b),
            });
        }
    }

    return {
        version: 'alpha-live-1',
        parent_path: опции.имя || 'мой звук',
        длительность_сек: +(audioBuffer.duration || 0).toFixed(2),
        atoms_count: атомы.length,
        atoms: атомы,
    };
}

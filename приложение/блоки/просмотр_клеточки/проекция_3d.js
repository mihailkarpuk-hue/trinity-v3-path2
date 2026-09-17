// Общий рендерер для проекций «Атомы» и «Решётка» (Three.js).
// Один canvas, один renderer/scene/camera/orbit; геометрию заменяем при смене клеточки.
// Поддержка временно́й развёртки: setVisibleByTime(t) показывает только
// атомы с birth ≤ t и связи, у которых оба конца уже видны.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function создать_3d_среду(контейнер_родитель, опции = {}) {
    const узел = document.createElement('div');
    узел.className = `проекция проекция-3d ${опции.класс || ''}`.trim();
    контейнер_родитель.appendChild(узел);

    const подсказка = document.createElement('div');
    подсказка.className = 'подсказка-вращения';
    подсказка.textContent = 'Перетащите мышью для вращения · колесо — приблизить';
    узел.appendChild(подсказка);

    const подсказка_осей = document.createElement('div');
    подсказка_осей.className = 'подсказка-осей';
    подсказка_осей.innerHTML =
        '<div><span class="ось-x">X</span> — время</div>' +
        '<div><span class="ось-y">Y</span> — log частоты</div>' +
        '<div><span class="ось-z">Z</span> — амплитуда</div>';
    узел.appendChild(подсказка_осей);

    const индикатор = document.createElement('div');
    индикатор.className = 'индикатор-загрузки';
    индикатор.textContent = '';
    индикатор.hidden = true;
    узел.appendChild(индикатор);

    const сцена = new THREE.Scene();
    сцена.background = new THREE.Color(0x000000);

    const камера = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    камера.position.set(3, 2, 5);

    const окружающий = new THREE.AmbientLight(0xffffff, 0.45);
    const направленный = new THREE.DirectionalLight(0xffffff, 0.7);
    направленный.position.set(5, 8, 5);
    сцена.add(окружающий, направленный);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    узел.appendChild(renderer.domElement);

    const controls = new OrbitControls(камера, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const группа = new THREE.Group();
    сцена.add(группа);

    let облако = null;             // Points
    let births_отсорт = null;       // Float32Array (упорядочены по росту birth)
    let рёбра_по_осям = [];         // [{LineSegments, thresholds:Float32Array}]
    let последнее_t = Infinity;     // показывать всё по умолчанию

    function resize() {
        const r = узел.getBoundingClientRect();
        const w = Math.max(1, Math.round(r.width));
        const h = Math.max(1, Math.round(r.height));
        renderer.setSize(w, h, false);
        камера.aspect = w / h;
        камера.updateProjectionMatrix();
    }
    const ro = new ResizeObserver(resize);
    ro.observe(узел);
    resize();

    let активна = false, raf = 0;
    function tick() {
        if (!активна) return;
        controls.update();
        renderer.render(сцена, камера);
        raf = requestAnimationFrame(tick);
    }

    function заменить_содержимое({ облако_новое, рёбра_новые, births_новые }) {
        // Чистим
        while (группа.children.length) {
            const o = группа.children.pop();
            o.geometry?.dispose?.();
            const m = o.material;
            if (Array.isArray(m)) m.forEach(x => x.dispose?.());
            else m?.dispose?.();
        }
        облако = null; рёбра_по_осям = []; births_отсорт = null;

        if (облако_новое) { группа.add(облако_новое); облако = облако_новое; }
        if (Array.isArray(рёбра_новые)) {
            for (const r of рёбра_новые) {
                группа.add(r.сегмент);
                рёбра_по_осям.push(r);
            }
        }
        births_отсорт = births_новые || null;
        вписать_камеру_в_группу();
        // По умолчанию — всё видно
        setVisibleByTime(Infinity);
    }

    function вписать_камеру_в_группу() {
        const box = new THREE.Box3().setFromObject(группа);
        if (box.isEmpty()) return;
        const центр = box.getCenter(new THREE.Vector3());
        const размер = box.getSize(new THREE.Vector3()).length() || 1;
        controls.target.copy(центр);
        камера.position.copy(центр).add(new THREE.Vector3(размер * 0.9, размер * 0.55, размер * 1.2));
        камера.near = размер / 100;
        камера.far  = размер * 100;
        камера.updateProjectionMatrix();
        controls.update();
    }

    function setVisibleByTime(t) {
        последнее_t = t;
        // Атомы: упорядочены по возрастанию birth, ищем индекс первого > t.
        if (облако && births_отсорт) {
            const N = upperBound(births_отсорт, t);
            облако.geometry.setDrawRange(0, N);
        }
        // Рёбра: thresholds[i] = max(birth_a, birth_b) для каждого сегмента;
        // сегменты отсортированы по threshold по возрастанию.
        // Каждое ребро занимает 2 вершины → setDrawRange(0, N*2).
        for (const r of рёбра_по_осям) {
            const M = upperBound(r.thresholds, t);
            r.сегмент.geometry.setDrawRange(0, M * 2);
        }
    }

    return {
        узел,
        активировать() {
            узел.classList.add('активна');
            активна = true;
            resize();
            raf = requestAnimationFrame(tick);
        },
        деактивировать() {
            узел.classList.remove('активна');
            активна = false;
            cancelAnimationFrame(raf);
        },
        заменить_содержимое,
        пометить_режим(имя_режима) {
            узел.classList.toggle('режим-физика', имя_режима === 'физика');
            узел.classList.toggle('режим-авто',   имя_режима === 'авто');
        },
        показать_индикатор(текст) { индикатор.textContent = текст || 'Загрузка…'; индикатор.hidden = false; },
        спрятать_индикатор()       { индикатор.hidden = true; },
        setVisibleByTime,
        снять() {
            ro.disconnect(); cancelAnimationFrame(raf);
            controls.dispose(); renderer.dispose(); узел.remove();
        },
    };
}

// === Бинарный поиск (последний индекс i, такой что arr[i] ≤ t) → возвращает COUNT видимых
function upperBound(arr, t) {
    if (!arr || !arr.length) return 0;
    if (!isFinite(t)) return arr.length;
    let lo = 0, hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (arr[mid] <= t) lo = mid + 1; else hi = mid;
    }
    return lo; // количество элементов ≤ t
}

// === Загрузка ===
export async function загрузить_атомы(url) {
    const ответ = await fetch(encodeURI(url), { cache: 'force-cache' });
    if (!ответ.ok) throw new Error('Атомы HTTP ' + ответ.status);
    return ответ.json();
}

// === Сборка содержимого с учётом синхронизации по времени ===
// режим_коорд:
//   'авто'    — pos_x/y/z из материнского генератора (сферическая раскладка)
//   'физика'  — X=birth (время), Y=log10(freq), Z=amp (амплитуда)
// Буква алфавита: частота → цвет (фиолет низ → красный верх), как в alphabet_atom.json.
function freqToHue(hz) {
    const t = Math.log10(Math.max(20, Math.min(hz, 20000)) / 20) / Math.log10(1000);
    return 270 - Math.min(t, 1) * 270;
}
function hslToRgb(h, s, l) {
    h /= 360; const a = s * Math.min(l, 1 - l);
    const f = (k) => { const x = (k + h * 12) % 12; return l - a * Math.max(-1, Math.min(x - 3, 9 - x, 1)); };
    return [f(0), f(8), f(4)];
}

export function собрать_контент(данные_атомов, режим /* 'атомы' | 'решётка' */, режим_коорд = 'авто', цвет_режим = 'файл') {
    const атомы_исх = данные_атомов?.atoms || [];
    const n = атомы_исх.length;
    if (!n) return null;

    // Сортируем по birth (стабильно)
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((i, j) => (атомы_исх[i].birth ?? 0) - (атомы_исх[j].birth ?? 0));
    const атомы = idx.map(i => атомы_исх[i]);

    // Позиции / цвета
    const есть_pos = атомы[0].pos_x !== undefined;
    const tmin = minKey(атомы, 'birth'), tmax = maxKey(атомы, 'birth');
    const fmin = Math.log10(Math.max(1, minKey(атомы, 'freq')));
    const fmax = Math.log10(Math.max(1, maxKey(атомы, 'freq')));
    const amax = Math.max(1e-6, maxKey(атомы, 'amp'));

    // Физика — масштабы для красивого габарита (~ ±1.5 по каждой оси)
    const X_SCALE = 1.8;  // время по горизонтали
    const Y_SCALE = 1.5;  // log частоты
    const Z_SCALE = 1.2;  // амплитуда

    const использовать_авто = (режим_коорд === 'авто' && есть_pos);

    const позиции = new Float32Array(n * 3);
    const цвета   = new Float32Array(n * 3);
    const births  = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        const a = атомы[i];
        let x, y, z;
        if (использовать_авто) {
            x = +a.pos_x || 0; y = +a.pos_y || 0; z = +a.pos_z || 0;
        } else if (режим_коорд === 'радиус') {
            // Радиальная раскладка: волна расходится ВО ВСЕ СТОРОНЫ (как у сцены-пульса).
            // Радиус растёт со временем (кольца от удара), угол — золотой спиральный шаг
            // (равномерно во все стороны независимо от частоты),
            // высота Y = log-частота (структура спектра остаётся).
            const угол = i * 2.399963229;                       // золотой угол
            const радиус = 0.30 + 2.5 * норм(a.birth, tmin, tmax);
            x = Math.cos(угол) * радиус;
            z = Math.sin(угол) * радиус;
            y = (норм(Math.log10(Math.max(1, a.freq)), fmin, fmax) - 0.5) * Y_SCALE;
        } else {
            // Физическая раскладка: время × log-частота × амплитуда
            x = (2 * норм(a.birth, tmin, tmax) - 1) * X_SCALE;
            y = (2 * норм(Math.log10(Math.max(1, a.freq)), fmin, fmax) - 1) * Y_SCALE;
            z = ((a.amp || 0) / amax) * Z_SCALE;
        }
        позиции[i*3] = x; позиции[i*3+1] = y; позиции[i*3+2] = z;
        if (цвет_режим === 'hz') {
            // Цвет = частота (Hz) — спектральный образ звука по алфавиту.
            const ampN = Math.min(1, (a.amp || 0) * 6);
            const [rC, gC, bC] = hslToRgb(freqToHue(a.freq || 20), 0.85, 0.45 + ampN * 0.4);
            цвета[i*3] = rC; цвета[i*3+1] = gC; цвета[i*3+2] = bC;
        } else {
            цвета[i*3]   = (a.color_r ?? 200) / 255;
            цвета[i*3+1] = (a.color_g ?? 200) / 255;
            цвета[i*3+2] = (a.color_b ?? 200) / 255;
        }
        births[i] = +a.birth || 0;
    }

    const геом = new THREE.BufferGeometry();
    геом.setAttribute('position', new THREE.BufferAttribute(позиции, 3));
    геом.setAttribute('color',    new THREE.BufferAttribute(цвета, 3));
    геом.setDrawRange(0, n);

    const мат = new THREE.PointsMaterial({
        size: 0.06,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.95,
    });
    const облако = new THREE.Points(геом, мат);

    let рёбра = null;
    if (режим === 'решётка') {
        рёбра = собрать_рёбра(атомы, позиции);
    }
    return { облако_новое: облако, рёбра_новые: рёбра, births_новые: births };
}

function собрать_рёбра(атомы_отсорт, позиции) {
    const n = атомы_отсорт.length;
    if (n < 2) return [];

    const оси = [
        { имя: 'time',     ключ: 'birth',          цвет: 0x4682B4 },
        { имя: 'freq',     ключ: 'freq',           цвет: 0x32CD32 },
        { имя: 'harmonic', ключ: 'harmonic_index', цвет: 0xFF6347 },
    ];
    const результат = [];

    for (const ось of оси) {
        // Сортируем индексы атомов (от 0..n-1) по ключу оси
        const idx = Array.from({ length: n }, (_, i) => i);
        idx.sort((i, j) => (атомы_отсорт[i][ось.ключ] ?? 0) - (атомы_отсорт[j][ось.ключ] ?? 0));

        // Пары последовательных по оси атомов
        const пары = new Array(idx.length - 1);
        for (let k = 0; k < idx.length - 1; k++) пары[k] = [idx[k], idx[k+1]];

        // threshold(пары) = max(birth_a, birth_b) — порог появления ребра
        for (const p of пары) {
            const a = атомы_отсорт[p[0]].birth ?? 0;
            const b = атомы_отсорт[p[1]].birth ?? 0;
            p.push(Math.max(a, b));
        }
        // Сортируем пары по порогу появления (по возрастанию)
        пары.sort((a, b) => a[2] - b[2]);

        const verts = new Float32Array(пары.length * 6);
        const thresholds = new Float32Array(пары.length);
        for (let k = 0; k < пары.length; k++) {
            const [a, b, th] = пары[k];
            verts[k*6  ] = позиции[a*3  ]; verts[k*6+1] = позиции[a*3+1]; verts[k*6+2] = позиции[a*3+2];
            verts[k*6+3] = позиции[b*3  ]; verts[k*6+4] = позиции[b*3+1]; verts[k*6+5] = позиции[b*3+2];
            thresholds[k] = th;
        }
        const геом = new THREE.BufferGeometry();
        геом.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        геом.setDrawRange(0, пары.length * 2);
        const мат = new THREE.LineBasicMaterial({
            color: ось.цвет, transparent: true, opacity: 0.35, depthWrite: false,
        });
        const сег = new THREE.LineSegments(геом, мат);
        сег.userData.ось = ось.имя;
        результат.push({ сегмент: сег, thresholds });
    }
    return результат;
}

function minKey(arr, key) { let m = Infinity; for (const a of arr) { const v = +a[key]; if (v < m) m = v; } return m === Infinity ? 0 : m; }
function maxKey(arr, key) { let m = -Infinity; for (const a of arr) { const v = +a[key]; if (v > m) m = v; } return m === -Infinity ? 0 : m; }
function норм(v, a, b) { if (b === a) return 0.5; return (v - a) / (b - a); }

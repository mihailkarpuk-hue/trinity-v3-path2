// Тринити Альфа — «Свой голос зримый».
// Путь: запиши/загрузи → атомизируй → нарисуй ТЕМ ЖЕ движком.
// Делает виртуальную клеточку (в памяти, без файлов на диске) и пускает её
// по той же шине событий (cellSelected), что и 218 готовых клеточек.

import { атомизировать } from './атомизация.js';

// Реестр пользовательских клеточек (в памяти браузера).
window.__пользовательские = window.__пользовательские || {
    атомы: new Map(),   // id → данные_атомов (как корпусный JSON: {atoms:[...]})
    звуки: new Map(),   // id → blob URL для <audio>
};
let счёт = 0;
let ctxДекод = null;     // отдельный AudioContext для decodeAudioData

function декод_контекст() {
    if (!ctxДекод) ctxДекод = new (window.AudioContext || window.webkitAudioContext)();
    return ctxДекод;
}

// ─── Показать атомизированный звук как клеточку ──────────────────────────
async function показать_буфер(audioBuffer, url, имя) {
    const данные = атомизировать(audioBuffer, { имя });
    if (!данные.atoms_count) { alert('Звук слишком тихий — атомов не нашлось.'); return; }

    const id = 'user_' + (++счёт);
    window.__пользовательские.атомы.set(id, данные);
    window.__пользовательские.звуки.set(id, url);

    const клеточка = {
        id,
        название: имя,
        группа: 'моё',
        звук: '',                 // файла на диске нет
        сцена: '', атомы: '', решётка: '',
        число_атомов: данные.atoms_count,
        число_связей: 0,
        длительность_сек: данные.длительность_сек,
        основной_элемент: 'мой',
        __в_памяти: true,
        __url: url,
    };
    window.__текущая_клеточка = { id, данные: клеточка };

    // У пользовательского звука нет «Сцены» — уводим на «Атомы».
    const кн_атомы = document.querySelector('.прж[data-проекция="атомы"]');
    if (кн_атомы && !кн_атомы.classList.contains('активна')) кн_атомы.click();

    // Пускаем по шине — просмотр нарисует, плеер подхватит звук.
    window.dispatchEvent(new CustomEvent('cellSelected', { detail: { id, данные: клеточка } }));

    обновить_паспорт(клеточка);
}

function обновить_паспорт(d) {
    const узел = document.getElementById('паспорт');
    if (!узел) return;
    const длит = (typeof d['длительность_сек'] === 'number') ? d['длительность_сек'].toFixed(1) : '?';
    const ат = (d['число_атомов'] ?? 0).toLocaleString('ru-RU');
    узел.textContent = `${d['название']} · ${длит} сек · ${ат} атомов · моё`;
}

async function blob_в_буфер(blob) {
    const arr = await blob.arrayBuffer();
    return await декод_контекст().decodeAudioData(arr);
}

// ─── Загрузка файла ──────────────────────────────────────────────────────
let файл_инпут = null;
function загрузить_файл() {
    if (!файл_инпут) {
        файл_инпут = document.createElement('input');
        файл_инпут.type = 'file';
        файл_инпут.accept = 'audio/*';
        файл_инпут.style.display = 'none';
        document.body.appendChild(файл_инпут);
        файл_инпут.addEventListener('change', async () => {
            const f = файл_инпут.files?.[0];
            файл_инпут.value = '';
            if (!f) return;
            try {
                const url = URL.createObjectURL(f);
                const буфер = await blob_в_буфер(f);
                await показать_буфер(буфер, url, f.name.replace(/\.[^.]+$/, ''));
            } catch (e) {
                console.error('[свой звук] файл:', e);
                alert('Не удалось прочитать аудиофайл: ' + e.message);
            }
        });
    }
    файл_инпут.click();
}

// ─── Запись с микрофона ──────────────────────────────────────────────────
let рекордер = null, чанки = [], индикатор = null;
async function записать_переключить() {
    if (рекордер && рекордер.state === 'recording') { рекордер.stop(); return; }
    let поток;
    try {
        поток = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
        alert('Микрофон недоступен: ' + e.message); return;
    }
    чанки = [];
    рекордер = new MediaRecorder(поток);
    рекордер.ondataavailable = (e) => { if (e.data.size) чанки.push(e.data); };
    рекордер.onstop = async () => {
        поток.getTracks().forEach(t => t.stop());
        спрятать_индикатор();
        try {
            const blob = new Blob(чанки, { type: чанки[0]?.type || 'audio/webm' });
            const url = URL.createObjectURL(blob);
            const буфер = await blob_в_буфер(blob);
            const имя = 'Запись ' + new Date().toLocaleTimeString('ru-RU').slice(0, 5);
            await показать_буфер(буфер, url, имя);
        } catch (e) {
            console.error('[свой звук] запись:', e);
            alert('Не удалось обработать запись: ' + e.message);
        }
    };
    рекордер.start();
    показать_индикатор();
}

function показать_индикатор() {
    if (!индикатор) {
        индикатор = document.createElement('div');
        индикатор.className = 'индикатор-записи';
        индикатор.innerHTML = `<span class="точка-записи"></span> Идёт запись… <button type="button" class="стоп-записи">Стоп</button>`;
        индикатор.querySelector('.стоп-записи').addEventListener('click', () => записать_переключить());
        document.body.appendChild(индикатор);
    }
    индикатор.hidden = false;
}
function спрятать_индикатор() { if (индикатор) индикатор.hidden = true; }

// ─── Подключение к меню «Функции» (data-id record/upload) ────────────────
function подключить_меню() {
    document.querySelectorAll('#панель-функций li').forEach((li) => {
        const id = li.dataset.id;
        if (id !== 'record' && id !== 'upload') return;
        li.addEventListener('click', (e) => {
            e.stopImmediatePropagation();           // глушим alert-заглушку из разметка.js
            if (id === 'record') записать_переключить();
            else загрузить_файл();
        }, true);                                   // capture: раньше обработчика-заглушки
    });
}

if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', подключить_меню);
else подключить_меню();

// Тринити Альфа — Блок 7 «Плеер звука»
// Один <audio>, один AudioContext + AnalyserNode (Решение 1, основной путь
// в части единого источника звука для проекций «Атомы» и «Решётка»).
// На «Сцене» включаем mute основного аудио и шлём шим в iframe сцены:
// внутри iframe есть свой АудиоКонтекст и live_analyzer — он играет сам,
// тогда наш <audio> молчит (двойного звука нет — спим основной).
// При переключении ОБРАТНО на «Атомы»/«Решётка» — пауза iframe-сцены
// (через клик по её внутренней кнопке Stop) + плей основного аудио.
//
// События в систему:
//   playbackTime    detail: {t}                  ← каждые ~50мс при Play
//   playbackStarted detail: {id}
//   playbackPaused  detail: {id}
//   playbackEnded   detail: {id}

const БАЗА_ДАННЫХ = '../../данные/клеточки/';

(() => {
    const зона = document.getElementById('зона-плеер');
    if (!зона) { console.error('[плеер] нет #зона-плеер'); return; }

    // 1) Чистим заглушку Задания 4
    зона.innerHTML = '';

    // 2) Аудио элемент + UI
    const аудио = document.createElement('audio');
    аудио.id = 'плеер-аудио';
    аудио.preload = 'auto';
    аудио.crossOrigin = 'anonymous';
    document.body.appendChild(аудио);

    const ui = document.createElement('div');
    ui.className = 'плеер-корень';
    ui.innerHTML = `
        <button class="плеер-кнопка" id="плеер-кн" type="button" aria-label="Воспроизвести">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="ico-play"><path d="M8 5v14l11-7z"/></svg>
        </button>
        <div class="плеер-полоса" id="плеер-полоса" role="slider"
             aria-label="Позиция" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" tabindex="0">
            <div class="плеер-полоса-заполнение" id="плеер-полоса-заполнение"></div>
            <div class="плеер-полоса-точка" id="плеер-полоса-точка"></div>
        </div>
        <div class="плеер-время" id="плеер-время">0:00 / 0:00</div>
    `;
    зона.appendChild(ui);

    const кн          = ui.querySelector('#плеер-кн');
    const полоса      = ui.querySelector('#плеер-полоса');
    const заполнение  = ui.querySelector('#плеер-полоса-заполнение');
    const точка       = ui.querySelector('#плеер-полоса-точка');
    const время_тэг   = ui.querySelector('#плеер-время');

    // 3) Web Audio (создаём ТОЛЬКО при первом Play — жест пользователя)
    let audioCtx = null, analyser = null, mediaSrc = null;
    function инициализировать_аудио_контекст() {
        if (audioCtx) return;
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.6;
        mediaSrc = audioCtx.createMediaElementSource(аудио);
        mediaSrc.connect(analyser);
        analyser.connect(audioCtx.destination);
        window.__плеер = { audioCtx, analyser, аудио };
    }

    // 4) Состояние
    let текущая_клеточка = null;
    let таймер_времени = 0;          // requestAnimationFrame id для шага времени

    // 5) Утилиты
    function м_сс(сек) {
        if (!isFinite(сек) || сек < 0) return '0:00';
        const m = Math.floor(сек / 60);
        const s = Math.floor(сек % 60);
        return m + ':' + (s < 10 ? '0' : '') + s;
    }
    function поставить_иконку(играет) {
        кн.innerHTML = играет
            ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>'
            : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
        кн.setAttribute('aria-label', играет ? 'Пауза' : 'Воспроизвести');
    }
    function обновить_полосу() {
        const t = аудио.currentTime || 0;
        const d = аудио.duration || 0;
        const p = d > 0 ? (t / d) : 0;
        заполнение.style.width = (p * 100).toFixed(2) + '%';
        точка.style.left       = (p * 100).toFixed(2) + '%';
        время_тэг.textContent  = м_сс(t) + ' / ' + м_сс(d);
        полоса.setAttribute('aria-valuenow', Math.round(p * 100));
    }

    // 6) Загрузка звука клеточки
    let synthUrl = null;
    function поставить_синтез(данные) {
        if (!данные?.atoms?.length || !window.__wav_url_из_атомов) return false;
        if (synthUrl) { try { URL.revokeObjectURL(synthUrl); } catch (e) { /* noop */ } }
        synthUrl = window.__wav_url_из_атомов(данные);
        аудио.src = synthUrl;
        return true;
    }
    function загрузить_клеточку(клеточка) {
        текущая_клеточка = клеточка;
        аудио.pause();
        поставить_иконку(false);
        if (клеточка?.['__в_памяти']) {
            аудио.src = клеточка['__url'];
            аудио.currentTime = 0;
            обновить_полосу();
            window.dispatchEvent(new CustomEvent('playbackTime', { detail: { t: 0 } }));
            return;
        }
        if (клеточка?.['звук'] && !клеточка['__звук_из_атомов']) {
            аудио.src = encodeURI(БАЗА_ДАННЫХ + клеточка['звук']);
            аудио.currentTime = 0;
            обновить_полосу();
            window.dispatchEvent(new CustomEvent('playbackTime', { detail: { t: 0 } }));
            return;
        }
        поставить_синтез(window.__текущие_атомы);
        обновить_полосу();
        window.dispatchEvent(new CustomEvent('playbackTime', { detail: { t: 0 } }));
    }
    аудио.addEventListener('error', () => {
        поставить_синтез(window.__текущие_атомы);
    });
    window.addEventListener('atomsReady', (e) => {
        const кл = window.__текущая_клеточка?.['данные'] || текущая_клеточка;
        if (кл?.['звук'] && !кл['__звук_из_атомов'] && аудио.src && !аудио.error) return;
        поставить_синтез(e.detail?.данные);
        обновить_полосу();
    });

    // 7) Цикл рассылки времени + уровней (50 мс)
    let таймер_50 = 0;
    function запустить_цикл_времени() {
        clearInterval(таймер_50);
        таймер_50 = setInterval(() => {
            const t = аудио.currentTime || 0;
            window.dispatchEvent(new CustomEvent('playbackTime', { detail: { t } }));
        }, 50);
    }
    function остановить_цикл_времени() {
        clearInterval(таймер_50);
        таймер_50 = 0;
    }

    // 8) Управление воспроизведением.
    // Единый источник звука — наш <audio> + анализатор. Обе проекции читают
    // analyser и идут синхронно со звуком: «Сцена» (реальная 3D-модель из
    // сцена_эталон.js) и «Образ» (из анализа спектра). iframe-сцен больше нет.
    async function play_pause() {
        if (!текущая_клеточка) return;
        инициализировать_аудио_контекст();
        if (audioCtx.state === 'suspended') await audioCtx.resume();
        if (аудио.paused) {
            // Safari не перематывает сам: если звук в конце — вернуть на начало, иначе play() = тишина
            if (аудио.ended || (аудио.duration && аудио.currentTime >= аудио.duration - 0.05)) аудио.currentTime = 0;
            try { await аудио.play(); }
            catch (e) { console.warn('[плеер] play() отвергнут:', e); return; }
            поставить_иконку(true);
            запустить_цикл_времени();
            window.dispatchEvent(new CustomEvent('playbackStarted', { detail: { id: текущая_клеточка.id } }));
        } else {
            аудио.pause();
            поставить_иконку(false);
            остановить_цикл_времени();
            window.dispatchEvent(new CustomEvent('playbackPaused', { detail: { id: текущая_клеточка.id } }));
        }
    }

    аудио.addEventListener('timeupdate', обновить_полосу);
    аудио.addEventListener('durationchange', обновить_полосу);
    аудио.addEventListener('loadedmetadata', обновить_полосу);
    аудио.addEventListener('ended', () => {
        поставить_иконку(false);
        остановить_цикл_времени();
        // Полностью раскрытое облако: финальный playbackTime = duration
        window.dispatchEvent(new CustomEvent('playbackTime',  { detail: { t: аудио.duration || 0 } }));
        window.dispatchEvent(new CustomEvent('playbackEnded', { detail: { id: текущая_клеточка?.id } }));
    });

    кн.addEventListener('click', play_pause);

    // 9) Перемотка по клику в полосе
    function перемотать_по_клику(e) {
        const r = полоса.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width;
        const доля = Math.max(0, Math.min(1, x));
        if (isFinite(аудио.duration) && аудио.duration > 0) {
            аудио.currentTime = доля * аудио.duration;
            обновить_полосу();
            window.dispatchEvent(new CustomEvent('playbackTime', { detail: { t: аудио.currentTime } }));
        }
    }
    полоса.addEventListener('click', перемотать_по_клику);

    // 10) Клавиатура (Решение 5.3)
    document.addEventListener('keydown', (e) => {
        // Игнорируем если пользователь печатает в инпуте
        const ц = e.target;
        const инпут = ц && (ц.tagName === 'INPUT' || ц.tagName === 'TEXTAREA' || ц.isContentEditable);
        if (инпут) return;

        if (e.key === ' ' || e.code === 'Space') {
            e.preventDefault();
            play_pause();
            return;
        }
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            if (window.__каталог_в_фокусе) return; // каталог листает клеточки
            if (!isFinite(аудио.duration) || аудио.duration <= 0) return;
            e.preventDefault();
            const шаг = (e.key === 'ArrowRight' ? 1 : -1) * 2;
            аудио.currentTime = Math.max(0, Math.min(аудио.duration - 0.01, (аудио.currentTime || 0) + шаг));
            обновить_полосу();
            window.dispatchEvent(new CustomEvent('playbackTime', { detail: { t: аудио.currentTime } }));
        }
    });

    // 11) Реакция на выбор клеточки в каталоге
    window.addEventListener('cellSelected', (e) => {
        const клеточка = e.detail?.['данные'];
        if (клеточка) загрузить_клеточку(клеточка);
    });

    // 12) Если каталог уже выбрал клеточку до нашего монтажа — подхватываем
    const уже = window.__текущая_клеточка?.['данные'];
    if (уже) загрузить_клеточку(уже);

    // 13) Звук — единый, на всех проекциях из нашего <audio>. Переключение
    //     вкладок не трогает воспроизведение: analyser продолжает питать и
    //     «Сцену», и «Образ» синхронно. (Mute/iframe-прокси удалены.)

    // Старт UI
    поставить_иконку(false);
    обновить_полосу();
})();

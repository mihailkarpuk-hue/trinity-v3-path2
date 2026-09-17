// Тринити Альфа — Блок 2 «Каркас» · логика каркаса
// 1) Переключатель проекций (Сцена / Атомы / Решётка) — визуальное.
// 2) Выпадающая панель «Функции» — открыть/закрыть, заглушки.

(function () {
    'use strict';

    // === 1. Переключатель проекций ===
    const кнопки_проекций = document.querySelectorAll('.прж');
    кнопки_проекций.forEach((кн) => {
        кн.addEventListener('click', () => {
            кнопки_проекций.forEach((other) => {
                other.classList.remove('активна');
                other.setAttribute('aria-selected', 'false');
            });
            кн.classList.add('активна');
            кн.setAttribute('aria-selected', 'true');
            // В Задании 4 контент не меняется — это будет в Задании 6.
        });
    });

    // === 2. Панель функций ===
    const кнопка = document.getElementById('кнопка-функции');
    const панель = document.getElementById('панель-функций');
    if (!кнопка || !панель) return;

    function открыть_закрыть(force) {
        const открыта = force !== undefined ? force : панель.hasAttribute('hidden');
        if (открыта) {
            панель.removeAttribute('hidden');
            кнопка.setAttribute('aria-expanded', 'true');
        } else {
            панель.setAttribute('hidden', '');
            кнопка.setAttribute('aria-expanded', 'false');
        }
    }

    кнопка.addEventListener('click', (e) => {
        e.stopPropagation();
        открыть_закрыть();
    });

    // Закрытие по клику вне панели
    document.addEventListener('click', (e) => {
        if (!панель.hasAttribute('hidden') &&
            !панель.contains(e.target) &&
            e.target !== кнопка) {
            открыть_закрыть(false);
        }
    });

    // Закрытие по Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !панель.hasAttribute('hidden')) {
            открыть_закрыть(false);
            кнопка.focus();
        }
    });

    // Пункты меню. record/upload обрабатывает модуль пользовательский_звук.js;
    // остальные — пока спокойный alert (Этап 2/3).
    панель.querySelectorAll('li').forEach((li) => {
        li.addEventListener('click', () => {
            открыть_закрыть(false);
            if (li.dataset.id === 'slovo') { window.open(encodeURI('../слово/index.html'), '_blank'); return; }
            if (li.dataset.id === 'faza') { window.open(encodeURI('../фаза_форма/index.html'), '_blank'); return; }
            if (['record','upload','hear','brick_sound','etalon','recognize'].includes(li.dataset.id)) return;
            const имя = li.textContent.trim();
            alert(`«${имя}» — эта функция будет добавлена в Этапе 2/3.`);
        });
    });
})();

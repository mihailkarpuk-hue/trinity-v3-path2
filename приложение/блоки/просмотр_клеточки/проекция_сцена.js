// Проекция «Сцена» — готовая Three.js-сцена материнского проекта в iframe.
// Способ встраивания изолирован: можно позднее заменить на inline-загрузку
// (Решение 1 — запасной путь), если мост postMessage окажется проблемным.

export function создать_проекцию_сцена(контейнер, опции = {}) {
    const узел = document.createElement('div');
    узел.className = 'проекция проекция-сцена';
    const iframe = document.createElement('iframe');
    iframe.title = 'Готовая 3D-сцена';
    iframe.referrerPolicy = 'no-referrer';
    iframe.allow = 'autoplay; xr-spatial-tracking';
    узел.appendChild(iframe);
    контейнер.appendChild(узел);

    let текущий_src = '';

    return {
        узел,
        активировать() { узел.classList.add('активна'); },
        деактивировать() { узел.classList.remove('активна'); },
        async обновить(клеточка) {
            // Путь к сцене из каталога. Кириллица — encodeURI (Решение 3).
            const путь = опции.базовый_путь + клеточка['сцена'];
            const url = encodeURI(путь);
            if (url === текущий_src) return;
            текущий_src = url;
            iframe.src = url;
        },
        // Для Задания 7 — мост postMessage к live_analyzer:
        отправить(payload) {
            try { iframe.contentWindow?.postMessage(payload, '*'); }
            catch (e) { /* noop */ }
        },
        // Прокси на внутренние контролы сцены. Та же origin → работает напрямую.
        нажать_в_сцене(селектор) {
            try {
                const doc = iframe.contentDocument || iframe.contentWindow?.document;
                const el = doc?.querySelector(селектор);
                if (el) { el.click(); return true; }
            } catch (e) { /* same-origin может быть недоступен после смены src */ }
            return false;
        },
        снять() { узел.remove(); },
    };
}

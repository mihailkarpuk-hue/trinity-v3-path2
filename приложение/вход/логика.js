// Тринити Альфа — Блок 1 «Вход» · логика перехода на главный экран

(function () {
    'use strict';

    const кнопка = document.getElementById('кнопка-войти');
    const перекрытие = document.getElementById('перекрытие');

    if (!кнопка || !перекрытие) return;

    // Путь к главному экрану. Кириллица — обязательно через encodeURI (Решение 3).
    const путь_к_главному = encodeURI('../главный_экран/index.html');

    let уже_перехожу = false;

    function войти() {
        if (уже_перехожу) return;
        уже_перехожу = true;
        перекрытие.classList.add('активно');
        // fade-out 500 ms — затем переход.
        setTimeout(() => {
            window.location.href = путь_к_главному;
        }, 500);
    }

    кнопка.addEventListener('click', войти);
    кнопка.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            войти();
        }
    });
})();

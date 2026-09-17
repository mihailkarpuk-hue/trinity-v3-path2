// Путь 2: слушаем организмы из выход/новое (библиотека моментов).
// Не использует словарь_кирпичей.json.

const КАТАЛОГ = "../../данные/организмы_каталог.json";
const WAV = "../../выход/новое/";

let текущий = null;
let текущаяКнопка = null;

function сброситьКнопку() {
  if (текущаяКнопка) {
    текущаяКнопка.classList.remove("играет");
    текущаяКнопка.textContent = "Слушать";
    текущаяКнопка = null;
  }
}

async function играть(рецепт, кнопка) {
  if (текущий) {
    текущий.pause();
    текущий = null;
    if (текущаяКнопка === кнопка) {
      сброситьКнопку();
      return;
    }
    сброситьКнопку();
  }
  const audio = new Audio(WAV + encodeURIComponent(рецепт.файл));
  текущий = audio;
  текущаяКнопка = кнопка;
  кнопка.classList.add("играет");
  кнопка.textContent = "Стоп";
  audio.addEventListener("ended", сброситьКнопку);
  try {
    await audio.play();
  } catch (e) {
    console.error(e);
    сброситьКнопку();
  }
}

async function main() {
  const данные = await fetch(КАТАЛОГ).then((r) => r.json());
  const список = document.getElementById("список");
  for (const р of данные.рецепты) {
    const li = document.createElement("li");
    li.className = "ряд";
    const info = document.createElement("div");
    const имя = document.createElement("p");
    имя.className = "имя";
    имя.textContent = р.имя;
    const состав = document.createElement("p");
    состав.className = "состав";
    состав.textContent = р.состав;
    info.append(имя, состав);
    const кн = document.createElement("button");
    кн.type = "button";
    кн.className = "кнопка";
    кн.textContent = "Слушать";
    кн.addEventListener("click", () => играть(р, кн));
    li.append(info, кн);
    список.append(li);
  }
}

main().catch(console.error);

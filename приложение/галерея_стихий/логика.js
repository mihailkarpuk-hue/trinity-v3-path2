// Галерея: живое видео (родной звук в файле) ↔ чистый образ.
// Правило: не вшиваем эталон — если в ролике есть дорожка, играет она.

const КАТАЛОГ_URL = "../../данные/стихии_живые/каталог.json";
const МЕДИА = "../../данные/стихии_живые/";

let каталог = null;
let текущий = null;
let audio = null;
let режим = "живое"; // живое | чистое
let текущийФайлВидео = null;

const $ = (id) => document.getElementById(id);

function медиаПуть(эл, файл) {
  return МЕДИА + эл.папка + "/" + encodeURIComponent(файл);
}

function видеоСоЗвукомВФайле(файл) {
  return typeof файл === "string" && файл.includes("video_live");
}

function стопВсё() {
  const v = $("видео");
  v.pause();
  v.muted = true;
  if (audio) {
    audio.pause();
    audio = null;
  }
  $("кн-играть").classList.remove("играет");
  $("кн-играть").textContent = "▶ Смотреть со звуком";
}

function подписьЖивого(эл, файл) {
  if (видеоСоЗвукомВФайле(файл) || эл.звук_из_видео) {
    return "живое видео · звук из ролика (не вшитый)";
  }
  return "живое видео · без дорожки (для ▶ нужен эталон)";
}

function обновитьИсточники() {
  if (!текущий) return;
  const эл = текущий;
  const video = $("видео");
  const img = $("фото");
  const видВыбор = $("выбор-видео");
  const фотВыбор = $("выбор-фото");
  видВыбор.innerHTML = "";
  фотВыбор.innerHTML = "";

  if (режим === "чистое") {
    const cv = эл.clean_video || "clean_video.mp4";
    const ci = эл.clean_image || "clean_image.png";
    текущийФайлВидео = cv;
    video.src = медиаПуть(эл, cv);
    video.loop = true;
    video.muted = true;
    video.load();
    if (эл.id === "veter") {
      $("подпись-видео").textContent = "схема потока (дым + линии) · ▶ = родной звук ролика";
    } else if (эл.звук_в_clean) {
      $("подпись-видео").textContent = "чистый образ · ▶ = звук из того же клипа (в такт)";
    } else {
      $("подпись-видео").textContent = "чистый образ (без фона) · движение из живого видео";
    }
    img.src = медиаПуть(эл, ci);
    img.alt = "чистый кадр · " + эл.имя;
    $("подпись-фото").textContent = "лучший чистый кадр";
  } else {
    let vids = [...(эл.видео || [])];
    // только live со звуком первыми; огонь — костёр video_02
    if (эл.id === "ogon" && vids.includes("video_02.mp4")) {
      vids = ["video_02.mp4", ...vids.filter((v) => v !== "video_02.mp4")];
    } else {
      const live = vids.filter(видеоСоЗвукомВФайле);
      const rest = vids.filter((v) => !видеоСоЗвукомВФайле(v));
      vids = live.length ? live : vids;
      // rest доступны только если live нет — иначе не путаем silent mixkit
      if (!live.length) vids = rest;
    }
    if (vids.length) {
      текущийФайлВидео = vids[0];
      video.src = медиаПуть(эл, vids[0]);
      video.loop = true;
      video.muted = true;
      video.load();
      vids.forEach((f, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "чип" + (i === 0 ? " активен" : "");
        const label =
          vids.length === 1
            ? видеоСоЗвукомВФайле(f)
              ? "со звуком"
              : "ролик"
            : видеоСоЗвукомВФайле(f)
              ? `со звуком ${i + 1}`
              : `ролик ${i + 1}`;
        b.textContent = label;
        b.addEventListener("click", () => {
          видВыбор.querySelectorAll(".чип").forEach((x) => x.classList.remove("активен"));
          b.classList.add("активен");
          стопВсё();
          текущийФайлВидео = f;
          video.src = медиаПуть(эл, f);
          video.load();
          $("подпись-видео").textContent = подписьЖивого(эл, f);
          обновитьСтатусЗвука();
        });
        видВыбор.append(b);
      });
    }
    $("подпись-видео").textContent = подписьЖивого(эл, текущийФайлВидео);
    const ph = эл.фото || [];
    if (ph.length) {
      img.src = медиаПуть(эл, ph[0]);
      img.alt = эл.имя;
      ph.forEach((f, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "чип" + (i === 0 ? " активен" : "");
        b.textContent = ph.length === 1 ? "фото" : `фото ${i + 1}`;
        b.addEventListener("click", () => {
          фотВыбор.querySelectorAll(".чип").forEach((x) => x.classList.remove("активен"));
          b.classList.add("активен");
          img.src = медиаПуть(эл, f);
        });
        фотВыбор.append(b);
      });
    }
    $("подпись-фото").textContent = "референс-фото";
  }

  video.onerror = () => {
    $("подпись-видео").textContent =
      режим === "чистое"
        ? "чистый слой ещё не собран — запусти scripts/чистый_образ_из_видео.py"
        : "видео не найдено";
  };
  обновитьСтатусЗвука();
}

function обновитьСтатусЗвука() {
  if (!текущий) return;
  const эл = текущий;
  if (режим === "чистое") {
    if (эл.звук_в_clean) {
      $("статус-звука").textContent = "чистый слой: ▶ = капли + звук из того же клипа (в такт)";
    } else if (эл.звук_из_видео) {
      $("статус-звука").textContent = "чистый слой: ▶ = схема + родной звук из живого ролика";
    } else if (эл.звук) {
      $("статус-звука").textContent = "чистый слой: ▶ = эталон корпуса";
    } else {
      $("статус-звука").textContent = "чистый слой без отдельного звука";
    }
    return;
  }
  if (видеоСоЗвукомВФайле(текущийФайлВидео) || эл.звук_из_видео) {
    $("статус-звука").textContent = "▶ = звук из видео (родной)";
  } else if (эл.звук) {
    $("статус-звука").textContent = "▶ = эталон корпуса (в ролике нет дорожки)";
  } else {
    $("статус-звука").textContent = "нет звука для этого ролика";
  }
}

function показать(эл) {
  стопВсё();
  текущий = эл;
  $("пусто").hidden = true;
  $("карточка").hidden = false;
  $("имя").textContent = эл.имя;

  const части = [];
  if (эл.клетка_id) части.push(эл.клетка_id);
  else части.push("нет клетки — только визуал");
  if (эл.звук_из_видео) части.push("звук из видео");
  else if (эл.звук) части.push("эталон *_real.wav");
  if (эл.заметка) части.push(эл.заметка);
  $("мета").textContent = части.join(" · ");

  $("кн-играть").disabled = false;
  обновитьИсточники();

  document.querySelectorAll(".пункт").forEach((p) => {
    p.classList.toggle("активен", p.dataset.id === эл.id);
  });
  history.replaceState(null, "", "#" + эл.id);
}

function файлРодногоЗвука(эл) {
  const vids = эл.видео || [];
  const live = vids.find((v) => typeof v === "string" && v.includes("video_live"));
  return live || vids[0] || null;
}

async function игратьСоЗвуком() {
  if (!текущий) return;
  const v = $("видео");
  const nativeLive = режим === "живое" && (видеоСоЗвукомВФайле(текущийФайлВидео) || текущий.звук_из_видео);
  // чистое со вшитой дорожкой того же клипа — идеальный такт капель
  const cleanEmbedded = режим === "чистое" && текущий.звук_в_clean;
  // чистое + родной звук отдельным файлом (ветер и т.п.)
  const nativeClean = режим === "чистое" && !cleanEmbedded && текущий.звук_из_видео && файлРодногоЗвука(текущий);

  if (!v.paused && $("кн-играть").classList.contains("играет")) {
    стопВсё();
    return;
  }
  стопВсё();
  v.currentTime = 0;

  if (nativeLive || cleanEmbedded) {
    v.muted = false;
    try {
      await v.play();
      $("статус-звука").textContent = cleanEmbedded
        ? "▶ чистый образ + звук того же клипа (в такт)"
        : "▶ звук из видео";
    } catch (e) {
      $("статус-звука").textContent = "видео: " + e.message;
    }
  } else if (nativeClean) {
    v.muted = true;
    try {
      await v.play();
    } catch (e) {
      $("статус-звука").textContent = "видео: " + e.message;
    }
    audio = new Audio(медиаПуть(текущий, файлРодногоЗвука(текущий)));
    audio.addEventListener("ended", () => {
      v.pause();
      стопВсё();
    });
    try {
      await audio.play();
      $("статус-звука").textContent = "▶ схема + родной звук ролика";
    } catch (e) {
      $("статус-звука").textContent = "звук: " + e.message;
    }
  } else {
    v.muted = true;
    try {
      await v.play();
    } catch (e) {
      $("статус-звука").textContent = "видео: " + e.message;
    }
    if (текущий.звук) {
      audio = new Audio(текущий.звук);
      audio.addEventListener("ended", () => {
        v.pause();
        стопВсё();
      });
      try {
        await audio.play();
        $("статус-звука").textContent = "▶ движение + звук эталона";
      } catch (e) {
        $("статус-звука").textContent = "звук: " + e.message;
      }
    }
  }
  $("кн-играть").classList.add("играет");
  $("кн-играть").textContent = "■ Стоп";
}

function построитьКаталог(элементы) {
  const nav = $("список");
  nav.innerHTML = "";
  for (const эл of элементы) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "пункт";
    b.dataset.id = эл.id;
    const mark = эл.звук_из_видео ? " · звук" : эл.clean_video ? " · clean" : "";
    b.innerHTML = `${эл.имя}<span class="мелкий">${эл.клетка_id || "без клетки"}${mark}</span>`;
    b.addEventListener("click", () => показать(эл));
    nav.append(b);
  }
}

function setРежим(r) {
  режим = r;
  document.querySelectorAll(".режим-кнопка").forEach((b) => {
    b.classList.toggle("активен", b.dataset.режим === r);
  });
  стопВсё();
  обновитьИсточники();
}

async function main() {
  каталог = await fetch(КАТАЛОГ_URL).then((r) => r.json());
  const элементы = [...каталог.элементы].sort((a, b) => a.порядок - b.порядок);
  построитьКаталог(элементы);

  $("кн-играть").addEventListener("click", игратьСоЗвуком);
  $("кн-стоп").addEventListener("click", стопВсё);
  $("р-живое").addEventListener("click", () => setРежим("живое"));
  $("р-чистое").addEventListener("click", () => setРежим("чистое"));

  const hash = (location.hash || "").replace(/^#/, "");
  const start = элементы.find((e) => e.id === hash) || элементы[0];
  if (start) показать(start);
}

main().catch((e) => {
  console.error(e);
  $("пусто").textContent = "Не удалось загрузить каталог.json";
});

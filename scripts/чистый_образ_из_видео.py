# -*- coding: utf-8 -*-
"""Чистый образ стихии из живого видео — без фона и лишнего.

Идея автора: сопоставить живое движение (видео) с чистым образом,
вырезанным из того же ролика. Звук для сопоставления — эталон корпуса
(*_real.wav); в галерее он играет синхронно с видео.

Алгоритм (детерминированный, без ML):
  огонь     — маска тёплых тонов HSV
  дождь     — яркие капли/пелена + движение между кадрами
  ветер     — движение (diff кадров), без статичного фона
  водопад   — яркая/холодная вода в движении
  гром/молния — вспышки яркости
  река      — движение воды (diff + сине-зелёная маска)

Выход на стихию:
  данные/стихии_живые/<id>/clean_video.mp4
  данные/стихии_живые/<id>/clean_image.png   (лучший кадр)
  данные/стихии_живые/<id>/clean_meta.json

Запуск: python3 scripts/чистый_образ_из_видео.py
        python3 scripts/чистый_образ_из_видео.py dozhd ogon
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

import cv2
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
БАЗА = os.path.join(КОРЕНЬ, "данные", "стихии_живые")
КАТ = os.path.join(БАЗА, "каталог.json")

# макс. секунд обработки (хватает для сопоставления; молния — дольше, вспышки редкие)
MAX_SEC = 8.0
MAX_SEC_DOZHD = 12.0
MAX_SEC_OGON = 12.0
MAX_SEC_MOLNIYA = 12.0
FFMPEG = os.path.join(КОРЕНЬ, "tools", "ffmpeg")


def _маска(stil: str, frame: np.ndarray, prev_gray: np.ndarray | None):
    """Бинарная маска «стихия» на чёрном."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)

    if stil == "ogon":
        # только пламя/угли — камни и кирпичи костра вырезаем
        warm1 = cv2.inRange(hsv, (0, 90, 90), (28, 255, 255))
        warm2 = cv2.inRange(hsv, (160, 90, 90), (180, 255, 255))
        yellow = cv2.inRange(hsv, (18, 80, 160), (40, 255, 255))
        mask = cv2.bitwise_or(warm1, warm2)
        mask = cv2.bitwise_or(mask, yellow)
        # серые камни + бежевые кирпичи
        rock = cv2.inRange(hsv, (0, 0, 15), (180, 70, 210))
        brick = cv2.inRange(hsv, (5, 20, 60), (30, 100, 220))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(rock))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(brick))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    elif stil == "dozhd":
        # дождь по воде: рябь/всплески (движение) + яркие короны; вода не плитка
        splash = cv2.threshold(v, 140, 255, cv2.THRESH_BINARY)[1]
        water = cv2.inRange(hsv, (70, 5, 20), (140, 160, 220))
        gray_water = cv2.inRange(hsv, (0, 0, 25), (180, 50, 180))
        waterish = cv2.bitwise_or(water, gray_water)
        if prev_gray is not None:
            d = cv2.absdiff(gray, prev_gray)
            mov = cv2.threshold(d, 5, 255, cv2.THRESH_BINARY)[1]
            soft = cv2.threshold(d, 3, 255, cv2.THRESH_BINARY)[1]
            dil = cv2.dilate(mov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            mask = cv2.bitwise_or(mov, soft)
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(splash, dil))
            # на водной поверхности оставляем слабую рябь
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(soft, waterish))
        else:
            mask = cv2.bitwise_and(splash, waterish)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.dilate(mask, k)
    elif stil == "veter":
        # дым/поток: серо-коричневая пелена + движение (дым медленный → мягкий порог)
        smoke = cv2.inRange(hsv, (0, 0, 35), (180, 70, 210))
        brown = cv2.inRange(hsv, (5, 15, 30), (35, 120, 180))
        mask = cv2.bitwise_or(smoke, brown)
        if prev_gray is not None:
            d = cv2.absdiff(gray, prev_gray)
            mov = cv2.threshold(d, 4, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.bitwise_or(mask, mov)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    elif stil == "vodopad":
        # только вода/пена — скалы (коричневые/серые тёмные) вырезаем
        water = cv2.inRange(hsv, (75, 10, 50), (140, 255, 255))
        foam = cv2.inRange(hsv, (0, 0, 170), (180, 55, 255))
        bright = cv2.threshold(v, 175, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_or(water, foam)
        mask = cv2.bitwise_or(mask, bright)
        # убрать коричнево-оранжевые скалы
        rock = cv2.inRange(hsv, (0, 25, 25), (35, 255, 210))
        gray_rock = cv2.inRange(hsv, (0, 0, 20), (180, 40, 140))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(rock))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(gray_rock))
        if prev_gray is not None:
            d = cv2.absdiff(gray, prev_gray)
            mov = cv2.threshold(d, 10, 255, cv2.THRESH_BINARY)[1]
            # движение только внутри водяной зоны (не по скалам)
            waterish = cv2.bitwise_or(water, foam)
            waterish = cv2.bitwise_or(waterish, bright)
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(mov, waterish))
    elif stil == "reka":
        water = cv2.inRange(hsv, (70, 15, 40), (140, 255, 255))
        white = cv2.inRange(hsv, (0, 0, 160), (180, 70, 255))
        mask = cv2.bitwise_or(water, white)
        if prev_gray is not None:
            d = cv2.absdiff(gray, prev_gray)
            mov = cv2.threshold(d, 8, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(mov, cv2.bitwise_or(water, white)))
    elif stil in ("grom", "molniya"):
        # тёмное небо + редкие вспышки: порог от перцентиля кадра, не абсолют
        mean_v = float(np.mean(v))
        p95 = float(np.percentile(v, 95))
        # вспышка = кадр заметно ярче обычного тёмного неба
        if mean_v < 12 and p95 < 40:
            # почти чёрный кадр — пусто (ждём вспышку)
            mask = np.zeros(gray.shape, np.uint8)
        else:
            thr = max(28, int(np.percentile(v, 88 if stil == "molniya" else 82)))
            mask = cv2.threshold(v, thr, 255, cv2.THRESH_BINARY)[1]
            # тонкие болты молнии — без OPEN (иначе съедает)
            if stil == "grom":
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            if prev_gray is not None:
                d = cv2.absdiff(gray, prev_gray)
                mov = cv2.threshold(d, 12, 255, cv2.THRESH_BINARY)[1]
                mask = cv2.bitwise_or(mask, mov)
            hh = mask.shape[0]
            mask[int(hh * 0.90):, :] = 0
            return mask, gray
    else:
        if prev_gray is None:
            return np.zeros(gray.shape, np.uint8), gray
        d = cv2.absdiff(gray, prev_gray)
        mask = cv2.threshold(d, 15, 255, cv2.THRESH_BINARY)[1]

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask, gray


def _выбрать_видео(эл: dict) -> str | None:
    """Предпочитаем video_live_* (со звуком); огонь — костёр video_02."""
    папка = os.path.join(БАЗА, эл["папка"])
    vids = list(эл.get("видео") or [])
    live = [v for v in vids if "video_live" in v]
    if live:
        vids = live + [v for v in vids if v not in live]
    if эл["id"] == "ogon" and "video_live_02.mp4" in vids:
        vids = ["video_live_02.mp4"] + [v for v in vids if v != "video_live_02.mp4"]
    elif эл["id"] == "ogon" and "video_02.mp4" in vids:
        vids = ["video_02.mp4"] + [v for v in vids if v != "video_02.mp4"]
    # ветер: #2 (дым/пожар) — лучшее движение потока → чистая схема с него
    if эл["id"] == "veter" and "video_live_02.mp4" in vids:
        vids = ["video_live_02.mp4"] + [v for v in vids if v != "video_live_02.mp4"]
    # дождь: live_01 = вода (озеро/пруд) + дождь → чистый слой + родной звук
    if эл["id"] == "dozhd" and "video_live_01.mp4" in vids:
        vids = ["video_live_01.mp4"] + [v for v in vids if v != "video_live_01.mp4"]
    for v in vids:
        p = os.path.join(папка, v)
        if os.path.isfile(p):
            return v
    # fallback: любой video_live / video_*.mp4 в папке
    for name in sorted(os.listdir(папка)):
        if name.startswith("video_live") and name.endswith((".mp4", ".webm")):
            return name
    return None


def обработать(эл: dict) -> dict | None:
    stil = эл["id"]
    папка = os.path.join(БАЗА, эл["папка"])
    src_name = _выбрать_видео(эл)
    if not src_name:
        print(f"  ✗ {stil}: нет видео")
        return None
    src = os.path.join(папка, src_name)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  ✗ не открыть {src}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if stil == "molniya":
        lim = MAX_SEC_MOLNIYA
    elif stil == "dozhd":
        lim = MAX_SEC_DOZHD
    elif stil == "ogon":
        lim = MAX_SEC_OGON
    else:
        lim = MAX_SEC
    max_frames = int(lim * fps)

    out_path = os.path.join(папка, "clean_video.mp4")
    silent_tmp = os.path.join(папка, "_clean_silent_tmp.mp4")
    # mp4v надёжнее на mac без ffmpeg; звук допишем ffmpeg из того же ролика
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_tmp, fourcc, fps, (w, h))

    prev = None
    best_frame = None
    best_score = -1
    n = 0
    scores = []

    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        mask, gray = _маска(stil, frame, prev)
        prev_gray = prev
        prev = gray
        clean = np.zeros_like(frame)
        if stil == "veter":
            # схема потока ветра: серая вуаль дыма + линии направления движения
            gray3 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smoke = cv2.cvtColor(gray3, cv2.COLOR_GRAY2BGR)
            clean[mask > 0] = smoke[mask > 0]
            clean = (clean.astype(np.float32) * 0.7).astype(np.uint8)
            if prev_gray is not None:
                # Farneback → редкие штрихи потока (схема, не фото)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                step = max(16, min(w, h) // 28)
                for y in range(step // 2, h, step):
                    for x in range(step // 2, w, step):
                        if mask[y, x] == 0:
                            continue
                        fx, fy = flow[y, x]
                        mag = float(np.hypot(fx, fy))
                        if mag < 1.2:
                            continue
                        x2 = int(x + fx * 2.2)
                        y2 = int(y + fy * 2.2)
                        cv2.line(clean, (x, y), (x2, y2), (210, 220, 230), 1, cv2.LINE_AA)
                        cv2.circle(clean, (x, y), 1, (180, 190, 200), -1)
        else:
            clean[mask > 0] = frame[mask > 0]
        writer.write(clean)

        score = float(np.count_nonzero(mask)) / (w * h)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_frame = clean.copy()
        n += 1

    cap.release()
    writer.release()

    img_path = os.path.join(папка, "clean_image.png")
    if best_frame is not None:
        cv2.imwrite(img_path, best_frame)
    else:
        img_path = None

    # родной звук из того же клипа → вшит в clean (такт капель = такт дорожки)
    звук_вшит = False
    if os.path.isfile(FFMPEG) and os.path.isfile(silent_tmp):
        # -shortest: длина = чистый слой; дорожка из исходника с t=0
        cmd = [
            FFMPEG, "-y",
            "-i", silent_tmp,
            "-i", src,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            out_path,
        ]
        r = __import__("subprocess").run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.isfile(out_path):
            звук_вшит = True
            try:
                os.remove(silent_tmp)
            except OSError:
                pass
        else:
            os.replace(silent_tmp, out_path)
    elif os.path.isfile(silent_tmp):
        os.replace(silent_tmp, out_path)

    meta = {
        "стихия": эл["имя"],
        "id": stil,
        "источник_видео": src_name,
        "кадров": n,
        "fps": fps,
        "сек": round(n / fps, 2),
        "доля_маски_med": round(float(np.median(scores)) if scores else 0, 4),
        "clean_video": "clean_video.mp4",
        "clean_image": "clean_image.png" if img_path else None,
        "звук_из_того_же_клипа": звук_вшит,
        "звук_для_сопоставления": эл.get("звук"),
        "правило": (
            "чистый слой из живого видео; звук — родная дорожка того же клипа (в такт)"
            if звук_вшит
            else "видео даёт движение; звук — эталон корпуса; чистый слой — маска без фона"
        ),
        "дата": date.today().isoformat(),
    }
    with open(os.path.join(папка, "clean_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  ✓ {stil}: {n} кадров → clean_video.mp4 + clean_image.png "
          f"(mask med={meta['доля_маски_med']})")
    return meta


def main() -> None:
    with open(КАТ, encoding="utf-8") as f:
        cat = json.load(f)
    want = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    элементы = sorted(cat["элементы"], key=lambda x: x["порядок"])
    print("чистый образ из живого видео →", БАЗА)
    done = []
    for эл in элементы:
        if want and эл["id"] not in want:
            continue
        print(эл["имя"], "…")
        m = обработать(эл)
        if m:
            done.append(m)
            # дописать в каталог флаги
            эл["clean_video"] = "clean_video.mp4"
            эл["clean_image"] = "clean_image.png"
            эл["звук_синхрон"] = True

    with open(КАТ, "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False, indent=2)

    # отчёт
    lines = [
        "# Чистые образы стихий из живого видео",
        "",
        f"> Дата: {date.today().isoformat()}",
        "> Звук для сопоставления = эталон `*_real.wav` (синхрон в галерее).",
        "> Чистый слой = стихия без фона (маска), не фото целиком.",
        "",
        "| Стихия | кадры | mask med | clean |",
        "|--------|------:|---------:|-------|",
    ]
    for m in done:
        lines.append(
            f"| {m['стихия']} | {m['кадров']} | {m['доля_маски_med']} | "
            f"`{m['id']}/clean_video.mp4` |"
        )
    lines += [
        "",
        "Галерея: `приложение/галерея_стихий/` — вкладки «живое» / «чистое» + звук эталона.",
        "Скрипт: `scripts/чистый_образ_из_видео.py`",
        "",
    ]
    otch = os.path.join(КОРЕНЬ, "отчёты", "чистые_образы_из_видео.md")
    with open(otch, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("отчёт →", otch)


if __name__ == "__main__":
    main()

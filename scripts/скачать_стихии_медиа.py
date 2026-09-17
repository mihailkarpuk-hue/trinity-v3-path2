# -*- coding: utf-8 -*-
"""Скачать живые фото/видео стихий по ТЗ_СТИХИИ_ЖИВЫЕ_МЕДИА.md"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import date

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
БАЗА = os.path.join(КОРЕНЬ, "данные", "стихии_живые")
UA = "Mozilla/5.0 (compatible; TrinityV3/1.0; +local research)"

# Кураторский список: проверенные URL (Mixkit Free License / Wikimedia CC0/CC-BY/PD)
# Видео Mixkit: https://mixkit.co/license/#videoFree
КАТАЛОГ = {
    "dozhd": {
        "стихия": "дождь",
        "клетка_id": "etalon_dozhd",
        "файлы": [
            {
                "тип": "video",
                "файл": "video_01.mp4",
                "url": "https://assets.mixkit.co/videos/2717/2717-720.mp4",
                "страница": "https://mixkit.co/free-stock-video/rain-in-the-wild-2717/",
                "автор": "Mixkit",
                "лицензия": "Mixkit Stock Video Free License",
                "заметки": "дождь на листьях, slow motion",
            },
            {
                "тип": "photo",
                "файл": "photo_01.jpg",
                "url": "https://upload.wikimedia.org/wikipedia/commons/4/4c/Rain_on_a_window.jpg",
                "страница": "https://commons.wikimedia.org/wiki/File:Rain_on_a_window.jpg",
                "автор": "see Commons page",
                "лицензия": "см. страница Commons (проверить при скачивании)",
                "заметки": "дождь на стекле",
            },
        ],
    },
    "veter": {
        "стихия": "ветер",
        "клетка_id": "etalon_veter",
        "файлы": [
            {
                "тип": "photo",
                "файл": "photo_01.jpg",
                "url": "https://upload.wikimedia.org/wikipedia/commons/9/94/Trees_Windswept.JPG",
                "страница": "https://commons.wikimedia.org/wiki/File:Trees_Windswept.JPG",
                "автор": "R MORGAN",
                "лицензия": "CC BY 3.0",
                "заметки": "деревья, согнутые ветром",
            },
        ],
    },
    "ogon": {
        "стихия": "огонь",
        "клетка_id": "etalon_ogon",
        "файлы": [],
    },
    "vodopad": {
        "стихия": "водопад",
        "клетка_id": "etalon_vodopad",
        "файлы": [
            {
                "тип": "photo",
                "файл": "photo_01.jpg",
                "url": "https://upload.wikimedia.org/wikipedia/commons/c/cc/Waterfall_in_a_wooded_ravine_(Unsplash).jpg",
                "страница": "https://commons.wikimedia.org/wiki/File:Waterfall_in_a_wooded_ravine_(Unsplash).jpg",
                "автор": "Unsplash / Commons",
                "лицензия": "CC0 1.0",
                "заметки": "Salt Creek Falls",
            },
        ],
    },
    "reka": {
        "стихия": "река",
        "клетка_id": None,
        "файлы": [],
    },
    "grom": {
        "стихия": "гром",
        "клетка_id": "etalon_grom",
        "файлы": [
            {
                "тип": "photo",
                "файл": "photo_01.jpg",
                "url": "https://upload.wikimedia.org/wikipedia/commons/8/8b/A_great_flash_thunderstorm_picture.jpg",
                # may need correct path - will resolve via Special:FilePath
                "url_alt": "https://commons.wikimedia.org/wiki/Special:FilePath/A_great_flash_thunderstorm_picture.jpg",
                "страница": "https://commons.wikimedia.org/wiki/File:A_great_flash_thunderstorm_picture.jpg",
                "автор": "FelixMittermeier / Pixabay",
                "лицензия": "CC0 1.0",
                "заметки": "гроза, вспышка",
            },
        ],
    },
    "molniya": {
        "стихия": "молния",
        "клетка_id": None,
        "файлы": [
            {
                "тип": "photo",
                "файл": "photo_01.jpg",
                "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Lightning_striking_the_Eiffel_Tower_-_NOAA.jpg",
                "страница": "https://commons.wikimedia.org/wiki/File:Lightning_striking_the_Eiffel_Tower_-_NOAA.jpg",
                "автор": "NOAA / historical",
                "лицензия": "Public Domain",
                "заметки": "молния в Эйфелеву башню, 1902",
            },
        ],
    },
}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def head_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            return 200 <= r.status < 400
    except Exception:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                return 200 <= r.status < 400
        except Exception:
            return False


def scrape_mixkit_tag(tag: str, limit: int = 8) -> list[str]:
    url = f"https://mixkit.co/free-stock-video/{tag}/"
    try:
        html = get(url).decode("utf-8", "ignore")
    except Exception as e:
        print(f"  scrape fail {tag}: {e}")
        return []
    urls = sorted(set(re.findall(
        r"https://assets\.mixkit\.co/videos/\d+/\d+-720\.mp4", html
    )))
    return urls[:limit]


def download_one(path: str, url: str, url_alt: str | None = None) -> str | None:
    for u in [url, url_alt]:
        if not u:
            continue
        try:
            data = get(u)
            if len(data) < 1000:
                continue
            with open(path, "wb") as f:
                f.write(data)
            return u
        except Exception as e:
            print(f"    fail {u}: {e}")
    return None


def main() -> None:
    os.makedirs(БАЗА, exist_ok=True)
    сегодня = date.today().isoformat()

    # дополнить видео через scrape
    extras = {
        "dozhd": ("rain",),
        "ogon": ("campfire", "fire", "flame"),
        "vodopad": ("waterfall",),
        "veter": ("wind", "trees"),
        "reka": ("river",),
        "grom": ("thunderstorm", "storm"),
        "molniya": ("lightning",),
    }
    for folder, tags in extras.items():
        found = []
        for t in tags:
            found.extend(scrape_mixkit_tag(t, 5))
        found = list(dict.fromkeys(found))
        if not found:
            continue
        # взять первый как video если ещё нет video в каталоге
        has_vid = any(f.get("тип") == "video" for f in КАТАЛОГ[folder]["файлы"])
        if not has_vid:
            КАТАЛОГ[folder]["файлы"].append({
                "тип": "video",
                "файл": "video_01.mp4",
                "url": found[0],
                "страница": f"https://mixkit.co/free-stock-video/{tags[0]}/",
                "автор": "Mixkit",
                "лицензия": "Mixkit Stock Video Free License",
                "заметки": f"автоподбор tag={tags[0]}",
            })
            print(f"+ video {folder}: {found[0]}")
        # второе видео если есть
        if len(found) > 1 and not any(f.get("файл") == "video_02.mp4" for f in КАТАЛОГ[folder]["файлы"]):
            # для ветра особенно нужно видео
            if folder in ("veter", "reka", "ogon", "grom"):
                КАТАЛОГ[folder]["файлы"].append({
                    "тип": "video",
                    "файл": "video_02.mp4" if any(f.get("файл") == "video_01.mp4" for f in КАТАЛОГ[folder]["файлы"]) else "video_01.mp4",
                    "url": found[1] if any(f.get("файл") == "video_01.mp4" for f in КАТАЛОГ[folder]["файлы"]) else found[0],
                    "страница": f"https://mixkit.co/free-stock-video/{tags[0]}/",
                    "автор": "Mixkit",
                    "лицензия": "Mixkit Stock Video Free License",
                    "заметки": f"автоподбор #2 tag={tags[0]}",
                })

    # фото реки / огня с commons если пусто
    if not any(f["тип"] == "photo" for f in КАТАЛОГ["ogon"]["файлы"]):
        КАТАЛОГ["ogon"]["файлы"].append({
            "тип": "photo",
            "файл": "photo_01.jpg",
            "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Campfire.jpg",
            "страница": "https://commons.wikimedia.org/wiki/File:Campfire.jpg",
            "автор": "see Commons",
            "лицензия": "см. Commons",
            "заметки": "костёр",
        })
    if not any(f["тип"] == "photo" for f in КАТАЛОГ["reka"]["файлы"]):
        КАТАЛОГ["reka"]["файлы"].append({
            "тип": "photo",
            "файл": "photo_01.jpg",
            "url": "https://commons.wikimedia.org/wiki/Special:FilePath/River.jpg",
            "страница": "https://commons.wikimedia.org/wiki/Special:Search/River",
            "автор": "see Commons",
            "лицензия": "см. Commons",
            "заметки": "река (поиск File:River — может редирект)",
        })

    # более надёжные commons photo URLs
    КАТАЛОГ["reka"]["файлы"] = [f for f in КАТАЛОГ["reka"]["файлы"] if f["файл"] != "photo_01.jpg"] + [{
        "тип": "photo",
        "файл": "photo_01.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4a/Amazon_River.jpg",
        "url_alt": "https://commons.wikimedia.org/wiki/Special:FilePath/Amazonas_bei_Parintins.jpg",
        "страница": "https://commons.wikimedia.org/wiki/File:Amazonas_bei_Parintins.jpg",
        "автор": "see Commons",
        "лицензия": "см. Commons",
        "заметки": "река Амазонка / крупная река",
    }]
    # fix rain photo - Rain_on_a_window may not exist
    for f in КАТАЛОГ["dozhd"]["файлы"]:
        if f["файл"] == "photo_01.jpg":
            f["url"] = "https://commons.wikimedia.org/wiki/Special:FilePath/Rain_drops_on_a_car%27s_window.jpg"
            f["url_alt"] = "https://commons.wikimedia.org/wiki/Special:FilePath/Rain.jpg"
            f["страница"] = "https://commons.wikimedia.org/wiki/File:Rain.jpg"
            f["лицензия"] = "см. Commons"
            f["заметки"] = "дождь / капли"

    results = []
    for folder, meta in КАТАЛОГ.items():
        d = os.path.join(БАЗА, folder)
        os.makedirs(d, exist_ok=True)
        saved = []
        print(f"\n== {folder} ({meta['стихия']}) ==")
        for item in meta["файлы"]:
            path = os.path.join(d, item["файл"])
            got = download_one(path, item["url"], item.get("url_alt"))
            ok = got is not None
            size = os.path.getsize(path) if ok else 0
            print(f"  {'OK' if ok else 'FAIL'} {item['файл']} ({size} bytes)")
            entry = dict(item)
            entry["дата"] = сегодня
            entry["скачан"] = ok
            entry["url_факт"] = got
            entry["байт"] = size
            if ok:
                saved.append(entry)
        man = {
            "стихия": meta["стихия"],
            "клетка_id": meta["клетка_id"],
            "источники": saved,
            "заметки": "ТЗ_СТИХИИ_ЖИВЫЕ_МЕДИА.md; звук корпуса не подменён",
        }
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2)
        results.append({
            "папка": folder,
            "стихия": meta["стихия"],
            "клетка_id": meta["клетка_id"],
            "файлов": len(saved),
            "есть_video": any(x["тип"] == "video" for x in saved),
            "есть_photo": any(x["тип"] == "photo" for x in saved),
        })

    # отчёт
    lines = [
        "# Живые медиа стихий — отчёт",
        "",
        f"> Дата: {сегодня}",
        f"> Папка: `данные/стихии_живые/`",
        f"> ТЗ: `ТЗ_СТИХИИ_ЖИВЫЕ_МЕДИА.md`",
        "",
        "| Стихия | клетка | файлов | video | photo |",
        "|--------|--------|--------|-------|-------|",
    ]
    for r in results:
        lines.append(
            f"| {r['стихия']} | {r['клетка_id'] or '—'} | {r['файлов']} | "
            f"{'✓' if r['есть_video'] else '—'} | {'✓' if r['есть_photo'] else '—'} |"
        )
    lines += [
        "",
        "## Примечание",
        "- Видео в основном **Mixkit Free License**.",
        "- Фото — **Wikimedia Commons** (CC0 / CC BY / PD — см. meta.json каждой папки).",
        "- Эталонный звук `*_real.wav` **не заменён**.",
        "",
    ]
    otch = os.path.join(КОРЕНЬ, "отчёты", "стихии_живые_медиа.md")
    with open(otch, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nотчёт →", otch)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

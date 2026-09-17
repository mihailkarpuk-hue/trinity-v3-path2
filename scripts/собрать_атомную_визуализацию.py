# -*- coding: utf-8 -*-
"""Собрать манифест + index для браузерной «Атомная визуализация».

Запуск: python3 scripts/собрать_атомную_визуализацию.py
"""
from __future__ import annotations

import json
import os
from datetime import date

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(КОРЕНЬ, "выход", "атомная_визуализация")
OUT_MANIFEST = os.path.join(OUT_DIR, "manifest.json")
OUT_HTML = os.path.join(OUT_DIR, "index.html")


def exists(*parts):
    return os.path.isfile(os.path.join(КОРЕНЬ, *parts))


def rel(*parts):
    return "/" + "/".join(parts)


def load_pkg(path):
    if not os.path.isfile(path):
        return {}
    return json.load(open(path, encoding="utf-8"))


def build_manifest():
    items = []

    # —— ДОЖДЬ live (PASS) ——
    if exists("выход", "атомы_полные_дождь_live", "атомы_звук_образ.json"):
        pkg = load_pkg(os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь_live", "атомы_звук_образ.json"))
        al = pkg.get("alignment") or {}
        items.append({
            "id": "dozhd_live",
            "имя": "Дождь",
            "статус": "PASS",
            "alignment": al.get("режим") or "single_live_clip",
            "n_atoms": pkg.get("n_atoms"),
            "событие": "drop_impact",
            "анализатор": True,
            "note": "PASS 2026-08-10 «отлично» · live_01 · drop_streak_impact_crown",
            "preview": rel("выход", "атомы_полные_дождь_live", "визуал", "preview.jpg") if exists("выход", "атомы_полные_дождь_live", "визуал", "preview.jpg") else None,
            "video_atoms": rel("выход", "атомы_полные_дождь_live", "визуал", "дождь_из_атомов.mp4"),
            "sound_atoms": rel("выход", "причина_дождь", "калибр_оси_live", "сборка_чистая.wav"),
            "sound_etalon": rel("выход", "причина_дождь", "live_sync", "dozhd_live_01_etalon.wav"),
            "compare": rel("выход", "причина_дождь", "сравнение_live", "рядом_честный.mp4") if exists("выход", "причина_дождь", "сравнение_live", "рядом_честный.mp4") else None,
            "still_live": rel("выход", "причина_дождь", "сравнение_live", "still_живое.jpg") if exists("выход", "причина_дождь", "сравнение_live", "still_живое.jpg") else None,
            "still_atoms": rel("выход", "причина_дождь", "сравнение_live", "still_атомы.jpg") if exists("выход", "причина_дождь", "сравнение_live", "still_атомы.jpg") else None,
            "E": [
                {"label": "ухо", "href": rel("выход", "причина_дождь", "E_дождь_single_live.html")},
                {"label": "сравнение", "href": rel("выход", "причина_дождь", "E_сравнение_живое_vs_атомы_live.html")},
                {"label": "глаз", "href": rel("выход", "атомы_полные_дождь_live", "E_визуал_из_атомов.html")},
            ],
            "accent": "#6a9fc9",
        })

    # —— ДОЖДЬ архив axes_decoupled ——
    pkg = load_pkg(os.path.join(КОРЕНЬ, "выход", "атомы_полные_дождь", "атомы_звук_образ.json"))
    items.append({
        "id": "dozhd_archive",
        "имя": "Дождь · архив 7350",
        "статус": "архив",
        "alignment": (pkg.get("alignment") or {}).get("режим") or "axes_decoupled",
        "n_atoms": pkg.get("n_atoms"),
        "событие": pkg.get("событие") or "drop_impact",
        "анализатор": False,
        "note": "архив axes_decoupled · sync — только live-пакет",
        "preview": rel("выход", "атомы_полные_дождь", "визуал", "preview_grid.jpg") if exists("выход", "атомы_полные_дождь", "визуал", "preview_grid.jpg") else None,
        "video_atoms": rel("выход", "атомы_полные_дождь", "визуал", "дождь_из_атомов.mp4"),
        "sound_atoms": rel("выход", "причина_дождь", "калибр_оси", "сборка_калибр_без_металла.wav") if exists("выход", "причина_дождь", "калибр_оси", "сборка_калибр_без_металла.wav") else rel("выход", "причина_дождь", "калибр_оси", "сборка_без_цикла.wav"),
        "sound_etalon": rel("выход", "причина_дождь", "калибр_оси", "эталон.wav") if exists("выход", "причина_дождь", "калибр_оси", "эталон.wav") else None,
        "compare": None,
        "E": [
            {"label": "глаз архив", "href": rel("выход", "атомы_полные_дождь", "E_визуал_из_атомов.html")},
        ],
        "accent": "#4a6a80",
    })

    # —— ГРОМ ——
    pkg = load_pkg(os.path.join(КОРЕНЬ, "выход", "атомы_полные_гром", "атомы_звук_образ.json"))
    items.append({
        "id": "grom",
        "имя": "Гром",
        "статус": "пилот",
        "alignment": (pkg.get("alignment") or {}).get("режим") or "axes_decoupled_flash",
        "n_atoms": pkg.get("n_atoms"),
        "событие": pkg.get("событие") or "flash_shockwave",
        "анализатор": False,
        "note": "долг sync molniya ≠ grom_real",
        "preview": rel("выход", "атомы_полные_гром", "визуал", "preview.jpg") if exists("выход", "атомы_полные_гром", "визуал", "preview.jpg") else None,
        "video_atoms": rel("выход", "атомы_полные_гром", "визуал", "гром_из_атомов.mp4"),
        "sound_atoms": rel("выход", "причина_гром", "калибр_оси", "сборка_104_чистая.wav") if exists("выход", "причина_гром", "калибр_оси", "сборка_104_чистая.wav") else rel("выход", "причина_гром", "калибр_оси", "сборка_без_цикла.wav"),
        "sound_etalon": rel("выход", "причина_гром", "калибр_оси", "эталон.wav") if exists("выход", "причина_гром", "калибр_оси", "эталон.wav") else None,
        "compare": None,
        "E": [
            {"label": "ухо 104", "href": rel("выход", "причина_гром", "E_гром_104_чистый.html")},
        ],
        "accent": "#c4b06a",
    })

    # —— ОГОНЬ ——
    pkg = load_pkg(os.path.join(КОРЕНЬ, "выход", "атомы_полные_огонь", "атомы_звук_образ.json"))
    al = pkg.get("alignment") or {}
    ts = pkg.get("trinity_sound_loop") or {}
    items.append({
        "id": "ogon",
        "имя": "Огонь",
        "статус": "PASS",
        "alignment": al.get("режим") or "single_live_clip",
        "n_atoms": pkg.get("n_atoms"),
        "событие": pkg.get("событие") or "combustion_crackle",
        "анализатор": True,
        "note": "PASS автора · v4 база снизу · синтез_из_атомов",
        "preview": rel("выход", "атомы_полные_огонь", "визуал", "preview.jpg"),
        "video_atoms": rel("выход", "атомы_полные_огонь", "визуал", "огонь_из_атомов.mp4"),
        "sound_atoms": rel("выход", "причина_огонь", "калибр_оси", "сборка_чистая.wav"),
        "sound_etalon": rel("выход", "причина_огонь", "калибр_оси", "эталон.wav") if exists("выход", "причина_огонь", "калибр_оси", "эталон.wav") else None,
        "compare": rel("выход", "причина_огонь", "сравнение", "рядом_честный_звук_атомы.mp4") if exists("выход", "причина_огонь", "сравнение", "рядом_честный_звук_атомы.mp4") else rel("выход", "причина_огонь", "сравнение", "рядом_звук_только_живое.mp4"),
        "still_live": rel("выход", "причина_огонь", "сравнение", "still_живое.jpg") if exists("выход", "причина_огонь", "сравнение", "still_живое.jpg") else None,
        "still_atoms": rel("выход", "причина_огонь", "сравнение", "still_атомы.jpg") if exists("выход", "причина_огонь", "сравнение", "still_атомы.jpg") else None,
        "E": [
            {"label": "сравнение", "href": rel("выход", "причина_огонь", "E_сравнение_живое_vs_атомы.html")},
            {"label": "ухо Тринити", "href": rel("выход", "причина_огонь", "E_огонь_тринити_честный.html")},
            {"label": "глаз", "href": rel("выход", "атомы_полные_огонь", "E_визуал_из_атомов.html")},
        ],
        "accent": "#e07840",
    })

    # —— ВЕТЕР ——
    pkg = load_pkg(os.path.join(КОРЕНЬ, "выход", "атомы_полные_ветер", "атомы_звук_образ.json"))
    al = pkg.get("alignment") or {}
    items.append({
        "id": "veter",
        "имя": "Ветер",
        "статус": "PASS",
        "alignment": al.get("режим") or "single_live_clip",
        "n_atoms": pkg.get("n_atoms"),
        "событие": pkg.get("событие") or "wind_stream_rustle",
        "анализатор": True,
        "note": "PASS ухо+глаз 2026-08-10 · broken_fiber_fleck · не дождь/река",
        "preview": rel("выход", "атомы_полные_ветер", "визуал", "preview.jpg"),
        "video_atoms": rel("выход", "атомы_полные_ветер", "визуал", "ветер_из_атомов.mp4"),
        "sound_atoms": rel("выход", "причина_ветер", "калибр_оси", "сборка_чистая.wav"),
        "sound_etalon": rel("выход", "причина_ветер", "live_sync", "veter_live_01_etalon.wav") if exists("выход", "причина_ветер", "live_sync", "veter_live_01_etalon.wav") else None,
        "compare": rel("выход", "причина_ветер", "сравнение", "рядом_честный.mp4") if exists("выход", "причина_ветер", "сравнение", "рядом_честный.mp4") else None,
        "still_live": rel("выход", "причина_ветер", "сравнение", "still_живое.jpg") if exists("выход", "причина_ветер", "сравнение", "still_живое.jpg") else None,
        "still_atoms": rel("выход", "причина_ветер", "сравнение", "still_атомы.jpg") if exists("выход", "причина_ветер", "сравнение", "still_атомы.jpg") else None,
        "E": [
            {"label": "ухо (шум атомы)", "href": rel("выход", "причина_ветер", "E_ветер_single_live.html")},
            {"label": "сравнение", "href": rel("выход", "причина_ветер", "E_сравнение_живое_vs_атомы.html")},
            {"label": "глаз", "href": rel("выход", "атомы_полные_ветер", "E_визуал_из_атомов.html")},
        ],
        "accent": "#8aa4b8",
    })

    # —— ВОДОПАД ——
    if exists("выход", "атомы_полные_водопад", "атомы_звук_образ.json"):
        pkg = load_pkg(os.path.join(КОРЕНЬ, "выход", "атомы_полные_водопад", "атомы_звук_образ.json"))
        al = pkg.get("alignment") or {}
        items.append({
            "id": "vodopad",
            "имя": "Водопад",
        "статус": "PASS",
        "alignment": al.get("режим") or "single_live_clip",
        "n_atoms": pkg.get("n_atoms"),
        "событие": "waterfall_fall_spray",
        "анализатор": True,
        "note": "PASS 2026-08-10 · Niagara Horseshoe",
            "preview": rel("выход", "атомы_полные_водопад", "визуал", "preview.jpg") if exists("выход", "атомы_полные_водопад", "визуал", "preview.jpg") else None,
            "video_atoms": rel("выход", "атомы_полные_водопад", "визуал", "водопад_из_атомов.mp4"),
            "sound_atoms": rel("выход", "причина_водопад", "калибр_оси", "сборка_чистая.wav"),
            "sound_etalon": rel("выход", "причина_водопад", "live_sync", "vodopad_niagara_etalon.wav") if exists("выход", "причина_водопад", "live_sync", "vodopad_niagara_etalon.wav") else rel("выход", "причина_водопад", "live_sync", "vodopad_live_01_etalon.wav"),
            "compare": rel("выход", "причина_водопад", "сравнение", "рядом_честный.mp4") if exists("выход", "причина_водопад", "сравнение", "рядом_честный.mp4") else None,
            "still_live": rel("выход", "причина_водопад", "сравнение", "still_живое.jpg") if exists("выход", "причина_водопад", "сравнение", "still_живое.jpg") else None,
            "still_atoms": rel("выход", "причина_водопад", "сравнение", "still_атомы.jpg") if exists("выход", "причина_водопад", "сравнение", "still_атомы.jpg") else None,
            "E": [
                {"label": "ухо", "href": rel("выход", "причина_водопад", "E_водопад_шум_атомы.html")},
                {"label": "сравнение", "href": rel("выход", "причина_водопад", "E_сравнение_живое_vs_атомы.html")},
                {"label": "глаз", "href": rel("выход", "атомы_полные_водопад", "E_визуал_из_атомов.html")},
            ],
            "accent": "#7a9eb0",
        })

    # —— РЕКА ——
    if exists("выход", "атомы_полные_река", "атомы_звук_образ.json"):
        pkg = load_pkg(os.path.join(КОРЕНЬ, "выход", "атомы_полные_река", "атомы_звук_образ.json"))
        al = pkg.get("alignment") or {}
        items.append({
            "id": "reka",
            "имя": "Река",
            "статус": "PASS",
            "alignment": al.get("режим") or "single_live_clip",
            "n_atoms": pkg.get("n_atoms"),
            "событие": "river_channel_flow",
            "анализатор": True,
            "note": "PASS глаз+ухо 2026-08-10 · current_filaments_foam",
            "preview": rel("выход", "атомы_полные_река", "визуал", "preview.jpg") if exists("выход", "атомы_полные_река", "визуал", "preview.jpg") else None,
            "video_atoms": rel("выход", "атомы_полные_река", "визуал", "река_из_атомов.mp4"),
            "sound_atoms": rel("выход", "причина_река", "калибр_оси", "сборка_чистая.wav"),
            "sound_etalon": rel("выход", "причина_река", "live_sync", "reka_live_01_etalon.wav"),
            "compare": rel("выход", "причина_река", "сравнение", "рядом_честный.mp4") if exists("выход", "причина_река", "сравнение", "рядом_честный.mp4") else None,
            "still_live": rel("выход", "причина_река", "сравнение", "still_живое.jpg") if exists("выход", "причина_река", "сравнение", "still_живое.jpg") else None,
            "still_atoms": rel("выход", "причина_река", "сравнение", "still_атомы.jpg") if exists("выход", "причина_река", "сравнение", "still_атомы.jpg") else None,
            "E": [
                {"label": "ухо", "href": rel("выход", "причина_река", "E_река_шум_атомы.html")},
                {"label": "сравнение", "href": rel("выход", "причина_река", "E_сравнение_живое_vs_атомы.html")},
                {"label": "глаз", "href": rel("выход", "атомы_полные_река", "E_визуал_из_атомов.html")},
            ],
            "accent": "#6a90a8",
        })

    return {
        "версия": "0.1",
        "имя": "Тринити · Атомная визуализация",
        "дата": date.today().isoformat(),
        "канон": "атом → кресты → клетка → синтез → образ",
        "правило": "один живой клип = звук + видео + образ (для новых стихий)",
        "элементы": items,
    }


HTML = r'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Тринити · Атомная визуализация</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg0:#07080a; --bg1:#10141a; --ink:#e6ebe8; --muted:#8a9399;
  --line:rgba(230,235,232,.12); --ok:#7cb89a; --wait:#c4a574;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg0);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.45}
body{
  min-height:100vh;
  background:
    radial-gradient(1200px 700px at 12% -10%, rgba(120,90,50,.18), transparent 55%),
    radial-gradient(900px 600px at 90% 0%, rgba(60,90,120,.16), transparent 50%),
    linear-gradient(180deg, #0b0d10 0%, #07080a 40%, #050607 100%);
}
a{color:inherit}
.wrap{max-width:1180px;margin:0 auto;padding:0 1.25rem 4rem}

/* HERO — one composition */
.hero{
  min-height:72vh;display:flex;flex-direction:column;justify-content:flex-end;
  padding:4.5rem 0 2.5rem;position:relative;
}
.brand{
  font-family:"Instrument Serif",Georgia,serif;
  font-size:clamp(2.6rem,7vw,4.8rem);letter-spacing:.02em;line-height:.95;
  margin:0 0 .6rem;font-weight:400;
}
.brand em{font-style:italic;color:#d8c4a4}
.tag{max-width:34rem;color:var(--muted);font-size:1.05rem;margin:0 0 1.6rem}
.cta{display:flex;gap:.7rem;flex-wrap:wrap}
.btn{
  appearance:none;border:1px solid var(--line);background:rgba(255,255,255,.03);
  color:var(--ink);padding:.7rem 1.1rem;border-radius:2px;cursor:pointer;
  font:500 .92rem "IBM Plex Sans",sans-serif;text-decoration:none;
}
.btn:hover{background:rgba(255,255,255,.07);border-color:rgba(230,235,232,.28)}
.btn.primary{background:rgba(216,196,164,.14);border-color:rgba(216,196,164,.35)}
.meta-bar{margin-top:2.2rem;display:flex;gap:1.5rem;flex-wrap:wrap;color:var(--muted);font-size:.82rem}
.meta-bar b{color:#cfd6d2;font-weight:500}

section{padding:2.5rem 0 0}
.sec-h{
  font-family:"Instrument Serif",Georgia,serif;font-size:1.85rem;font-weight:400;
  margin:0 0 .35rem;
}
.sec-p{color:var(--muted);margin:0 0 1.4rem;max-width:40rem}

.rail{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.2rem}
.chip{
  border:1px solid var(--line);padding:.45rem .85rem;border-radius:999px;
  background:transparent;color:var(--muted);cursor:pointer;font:500 .85rem inherit;
}
.chip.on{color:var(--ink);border-color:rgba(230,235,232,.35);background:rgba(255,255,255,.05)}

.stage{
  display:grid;grid-template-columns:1.15fr .85fr;gap:1.25rem;
  border-top:1px solid var(--line);padding-top:1.25rem;
}
@media (max-width:900px){.stage{grid-template-columns:1fr}}

.panel{
  background:linear-gradient(160deg, rgba(255,255,255,.035), rgba(255,255,255,.01));
  border:1px solid var(--line);border-radius:4px;overflow:hidden;
}
.panel .hd{padding:.85rem 1rem;border-bottom:1px solid var(--line);
  display:flex;justify-content:space-between;align-items:baseline;gap:1rem}
.panel .hd h3{margin:0;font-size:1rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.panel .hd .st{font-size:.78rem;color:var(--muted)}
.panel .bd{padding:1rem}
video,img{width:100%;display:block;background:#000;border-radius:2px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:.75rem}
@media (max-width:600px){.row2{grid-template-columns:1fr}}
.lbl{font-size:.75rem;color:var(--muted);margin:.35rem 0 .25rem}
audio{width:100%;margin:.15rem 0 .7rem;height:2.2rem}
.facts{display:grid;grid-template-columns:1fr 1fr;gap:.55rem .9rem;margin:0 0 1rem}
.facts div{border-bottom:1px solid var(--line);padding-bottom:.45rem}
.facts span{display:block;font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.facts b{font-weight:500;font-size:.95rem}
.links{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.4rem}
.links a{
  font-size:.82rem;padding:.4rem .7rem;border:1px solid var(--line);border-radius:2px;
  text-decoration:none;color:var(--muted);
}
.links a:hover{color:var(--ink);border-color:rgba(230,235,232,.3)}
.note{font-size:.88rem;color:var(--muted);margin:0 0 .9rem}
.badge{
  display:inline-block;font-size:.72rem;padding:.15rem .45rem;border-radius:2px;
  border:1px solid currentColor;letter-spacing:.04em;
}
.badge.pass{color:var(--ok)} .badge.e{color:var(--wait)} .badge.pilot{color:var(--muted)}
.hidden{display:none !important}
.foot{margin-top:3rem;padding-top:1.2rem;border-top:1px solid var(--line);
  color:var(--muted);font-size:.8rem}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <h1 class="brand">Тринити<br/><em>Атомная визуализация</em></h1>
    <p class="tag">Звук природы → атомы → обратный синтез и образ причины. Один клип — одна ось времени.</p>
    <div class="cta">
      <a class="btn primary" href="#сцена">Открыть стихии</a>
      <a class="btn" href="/выход/атомная_визуализация/E_все_живое_vs_атомы.html">Живое vs атомы · все</a>
      <a class="btn" href="/выход/атомная_визуализация/E_алфавит_пересечение.html">Алфавит ∩</a>
      <a class="btn" href="/выход/атомная_визуализация/E_кросс_pca.html">Кросс PCA</a>
      <a class="btn" href="/выход/атомная_визуализация/E_law_vs_instance.html">Law vs instance</a>
      <a class="btn" href="/выход/причина_ветер/E_ветер_single_live.html">Ветер · E ухо</a>
      <a class="btn" href="/выход/причина_огонь/E_сравнение_живое_vs_атомы.html">Огонь · PASS</a>
    </div>
    <div class="meta-bar" id="meta-bar"></div>
  </header>

  <section id="сцена">
    <h2 class="sec-h">Стихии</h2>
    <p class="sec-p">Выбери элемент: живое рядом с образом из атомов, звук эталона и сборки.</p>
    <div class="rail" id="rail"></div>
    <div class="stage" id="stage"></div>
  </section>

  <footer class="foot" id="foot"></footer>
</div>
<script>
const MANIFEST_URL = './manifest.json?v=' + Date.now();

function badge(st){
  const c = st==='PASS'?'pass':(st==='E'?'e':'pilot');
  return `<span class="badge ${c}">${st}</span>`;
}

function el(html){
  const t=document.createElement('template'); t.innerHTML=html.trim(); return t.content.firstChild;
}

async function main(){
  const m = await fetch(MANIFEST_URL).then(r=>r.json());
  document.getElementById('meta-bar').innerHTML =
    `<span><b>${m.элементы.length}</b> стихии</span>`+
    `<span>канон · <b>${m.канон}</b></span>`+
    `<span>сборка · <b>${m.дата}</b> · v${m.версия}</span>`;
  document.getElementById('foot').textContent =
    m.имя + ' · ' + m.правило;

  const rail = document.getElementById('rail');
  const stage = document.getElementById('stage');
  let active = m.элементы.find(x=>x.статус==='E')?.id || m.элементы[0].id;

  function renderRail(){
    rail.innerHTML='';
    m.элементы.forEach(e=>{
      const b=el(`<button class="chip ${e.id===active?'on':''}" type="button">${e.имя}</button>`);
      b.style.setProperty('--a', e.accent||'#aaa');
      if(e.id===active) b.style.borderColor = e.accent;
      b.onclick=()=>{active=e.id; renderRail(); renderStage()};
      rail.appendChild(b);
    });
  }

  function media(src, kind){
    if(!src) return '<p class="note">нет файла</p>';
    if(kind==='video') return `<video controls playsinline preload="metadata" src="${src}"></video>`;
    if(kind==='img') return `<img src="${src}" alt=""/>`;
    return `<audio controls preload="metadata" src="${src}"></audio>`;
  }

  function renderStage(){
    const e = m.элементы.find(x=>x.id===active);
    const stills = (e.still_live && e.still_atoms) ? `
      <div class="row2">
        <div><div class="lbl">живое</div>${media(e.still_live,'img')}</div>
        <div><div class="lbl">из атомов</div>${media(e.still_atoms,'img')}</div>
      </div>` : (e.preview?`<div class="lbl">превью атомов</div>${media(e.preview,'img')}`:'');

    const mainVid = e.compare || e.video_atoms;
    const mainLabel = e.compare ? 'рядом · живое | атомы' : 'образ из атомов';

    stage.innerHTML = `
      <div class="panel">
        <div class="hd"><h3>${mainLabel}</h3><span class="st">${badge(e.статус)}</span></div>
        <div class="bd">
          ${media(mainVid,'video')}
          ${stills}
          ${e.compare && e.video_atoms ? `<div class="lbl" style="margin-top:1rem">только атомы</div>${media(e.video_atoms,'video')}`:''}
        </div>
      </div>
      <div class="panel">
        <div class="hd"><h3>${e.имя}</h3><span class="st" style="color:${e.accent}">${e.событие||''}</span></div>
        <div class="bd">
          <p class="note">${e.note||''}</p>
          <div class="facts">
            <div><span>атомы</span><b>${e.n_atoms ?? '—'}</b></div>
            <div><span>alignment</span><b>${e.alignment||'—'}</b></div>
            <div><span>анализатор</span><b>${e.анализатор?'да':'нет / долг'}</b></div>
            <div><span>статус</span><b>${e.статус}</b></div>
          </div>
          <div class="lbl">звук · эталон</div>
          ${media(e.sound_etalon,'audio')}
          <div class="lbl">звук · из атомов</div>
          ${media(e.sound_atoms,'audio')}
          <div class="links">${(e.E||[]).map(x=>`<a href="${x.href}">${x.label}</a>`).join('')}</div>
        </div>
      </div>`;
  }

  renderRail(); renderStage();
}
main().catch(err=>{
  document.getElementById('stage').innerHTML = '<p class="note">Не удалось загрузить manifest.json · '+err+'</p>';
});
</script>
</body>
</html>
'''


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    man = build_manifest()
    # fix typo key for rain
    for e in man["элементы"]:
        if "анализator" in e:
            e["анализатор"] = e.pop("анализator")
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(HTML)
    # convenience root redirect page
    root = os.path.join(КОРЕНЬ, "атомная_визуализация.html")
    with open(root, "w", encoding="utf-8") as f:
        f.write(
            "<!DOCTYPE html><meta charset='utf-8'/>"
            "<meta http-equiv='refresh' content='0;url=/выход/атомная_визуализация/index.html'/>"
            "<p><a href='/выход/атомная_визуализация/index.html'>Тринити · Атомная визуализация</a></p>"
        )
    print(json.dumps({
        "ok": True,
        "url": "/выход/атомная_визуализация/",
        "элементы": [e["id"] + ":" + e["статус"] for e in man["элементы"]],
        "дата": man["дата"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

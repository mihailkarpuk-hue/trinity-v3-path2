# -*- coding: utf-8 -*-
"""Для каждой стихии: реальное видео+звук vs сборка из атомов (видео+звук).

По очереди: дождь_live → огонь → ветер → водопад → река → гром (с оговоркой axes).

Запуск: python3 scripts/собрать_сравнения_живое_vs_атомы.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import cv2

КОРЕНЬ = Path(__file__).resolve().parents[1]
FF = КОРЕНЬ / "tools" / "ffmpeg"
FFMPEG = str(FF if FF.is_file() else "ffmpeg")
HUB = КОРЕНЬ / "выход" / "атомная_визуализация" / "E_все_живое_vs_атомы.html"
REPORT = КОРЕНЬ / "отчёты" / "сравнения_живое_vs_атомы.json"

# порядок — как просил автор
STYLES = [
    {
        "id": "dozhd_live",
        "имя": "Дождь",
        "pkg": "выход/атомы_полные_дождь_live/атомы_звук_образ.json",
        "vis": "выход/атомы_полные_дождь_live/визуал/дождь_из_атомов.mp4",
        "wav": "выход/причина_дождь/калибр_оси_live/сборка_чистая.wav",
        "out": "выход/причина_дождь/сравнение_live",
        "e": "выход/причина_дождь/E_живое_vs_атомы.html",
    },
    {
        "id": "ogon",
        "имя": "Огонь",
        "pkg": "выход/атомы_полные_огонь/атомы_звук_образ.json",
        "vis": "выход/атомы_полные_огонь/визуал/огонь_из_атомов.mp4",
        "wav": "выход/причина_огонь/калибр_оси/сборка_чистая.wav",
        "out": "выход/причина_огонь/сравнение_av",
        "e": "выход/причина_огонь/E_живое_vs_атомы.html",
    },
    {
        "id": "veter",
        "имя": "Ветер",
        "pkg": "выход/атомы_полные_ветер/атомы_звук_образ.json",
        "vis": "выход/атомы_полные_ветер/визуал/ветер_из_атомов.mp4",
        "wav": "выход/причина_ветер/калибр_оси/сборка_чистая.wav",
        "out": "выход/причина_ветер/сравнение_av",
        "e": "выход/причина_ветер/E_живое_vs_атомы.html",
    },
    {
        "id": "vodopad",
        "имя": "Водопад",
        "pkg": "выход/атомы_полные_водопад/атомы_звук_образ.json",
        "vis": "выход/атомы_полные_водопад/визуал/водопад_из_атомов.mp4",
        "wav": "выход/причина_водопад/калибр_оси/сборка_чистая.wav",
        "out": "выход/причина_водопад/сравнение_av",
        "e": "выход/причина_водопад/E_живое_vs_атомы.html",
    },
    {
        "id": "reka",
        "имя": "Река",
        "pkg": "выход/атомы_полные_река/атомы_звук_образ.json",
        "vis": "выход/атомы_полные_река/визуал/река_из_атомов.mp4",
        "wav": "выход/причина_река/калибр_оси/сборка_чистая.wav",
        "out": "выход/причина_река/сравнение_av",
        "e": "выход/причина_река/E_живое_vs_атомы.html",
    },
    {
        "id": "grom",
        "имя": "Гром",
        "pkg": "выход/атомы_полные_гром/атомы_звук_образ.json",
        "vis": "выход/атомы_полные_гром/визуал/гром_из_атомов.mp4",
        "wav": "выход/причина_гром/калибр_оси/сборка_104_чистая.wav",
        "out": "выход/причина_гром/сравнение_av",
        "e": "выход/причина_гром/E_живое_vs_атомы.html",
        "note": "axes_decoupled_flash: вспышка molniya ≠ sync с grom_real",
    },
]


def run(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        err = (r.stderr or b"").decode("utf-8", "ignore")[-500:]
        raise RuntimeError(f"cmd fail {cmd[:4]}…\n{err}")
    return r


def resolve_axes(pkg: dict, style: dict):
    al = pkg.get("alignment") or {}
    if not isinstance(al, dict):
        al = {"режим": al}
    clip = al.get("clip") or al.get("video_ось")
    eta = al.get("звук_ось")
    t0 = al.get("t0_sec")
    dur = al.get("seg_dur_sec")

    # гром: вспышка отдельно
    if style["id"] == "grom":
        clip = al.get("вспышка_ось") or "данные/стихии_живые/molniya/video_live_01.mp4"
        eta = eta or "данные/клеточки/эталоны/grom_real.wav"
        t0 = 0.0 if t0 is None else t0
        # короткий кусок вспышки
        dur = 2.6 if dur is None else dur

    if not clip or not (КОРЕНЬ / clip).is_file():
        raise FileNotFoundError(f"нет клипа для {style['id']}: {clip}")
    if not eta or not (КОРЕНЬ / eta).is_file():
        raise FileNotFoundError(f"нет эталона звука для {style['id']}: {eta}")
    if t0 is None:
        t0 = 0.0
    if dur is None:
        dur = 2.6
    return str(КОРЕНЬ / clip), str(КОРЕНЬ / eta), float(t0), float(dur), al.get("режим")


def build_one(style: dict) -> dict:
    pkg = json.load(open(КОРЕНЬ / style["pkg"], encoding="utf-8"))
    clip, eta, t0, dur, режим = resolve_axes(pkg, style)
    vis = КОРЕНЬ / style["vis"]
    wav = КОРЕНЬ / style["wav"]
    if not vis.is_file():
        raise FileNotFoundError(vis)
    if not wav.is_file():
        raise FileNotFoundError(wav)

    out = КОРЕНЬ / style["out"]
    out.mkdir(parents=True, exist_ok=True)
    live_silent = out / "_live_silent.mp4"
    live_av = out / "A_живое_видео_звук.mp4"
    atom_av = out / "B_из_атомов_видео_звук.mp4"
    side = out / "рядом_живое_vs_атомы.mp4"
    still_a = out / "still_живое.jpg"
    still_b = out / "still_атомы.jpg"

    print(f"→ {style['имя']} t0={t0} dur={dur}", flush=True)

    run([
        FFMPEG, "-y", "-ss", str(t0), "-i", clip, "-t", str(dur),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an",
        str(live_silent),
    ])
    run([
        FFMPEG, "-y", "-i", str(live_silent), "-i", eta,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(live_av),
    ])
    run([
        FFMPEG, "-y", "-i", str(vis), "-i", str(wav), "-t", str(dur),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(atom_av),
    ])

    ca, cb = cv2.VideoCapture(str(live_av)), cv2.VideoCapture(str(atom_av))
    fps = ca.get(cv2.CAP_PROP_FPS) or 30.0
    n = max(1, int(dur * fps))
    wr = cv2.VideoWriter(str(side), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))
    for i in range(n):
        ok1, f1 = ca.read()
        ok2, f2 = cb.read()
        if not ok1 or not ok2:
            break
        f1 = cv2.resize(f1, (640, 360))
        f2 = cv2.resize(f2, (640, 360))
        wr.write(cv2.hconcat([f1, f2]))
        if i == min(12, n - 1):
            cv2.imwrite(str(still_a), f1)
            cv2.imwrite(str(still_b), f2)
    wr.release()
    ca.release()
    cb.release()

    note = style.get("note") or ""
    if режим and "decoupled" in str(режим):
        note = (note + " · " if note else "") + f"alignment={режим} (не single_live)"

    # relative urls from E page (same folder parent = причина_*)
    e_path = КОРЕНЬ / style["e"]
    rel_out = Path(style["out"]).name  # сравнение_av or сравнение_live
    e_path.write_text(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — живое vs атомы · {style['имя']}</title>
<style>
body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:1100px;margin:1.5rem auto;padding:0 1rem}}
video{{width:100%;background:#000;margin:.3rem 0 .8rem}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}}
img{{width:100%;background:#000;border-radius:6px}}
.meta{{opacity:.85}} a{{color:#9dceb0}}
</style></head><body>
<h1>{style['имя']}: реальное · рядом из атомов</h1>
<p class="meta">t0={t0}с · окно {dur}с · {режим or '—'}
{('<br/>⚠ ' + note) if note else ''}<br/>
<b>Слева</b> — живое видео + эталонный звук оси. <b>Справа</b> — визуал из атомов + звук из атомов.</p>
<div class="grid">
<div>
<p><b>A. Реальное</b></p>
<img src="{rel_out}/still_живое.jpg"/>
<video controls src="{rel_out}/A_живое_видео_звук.mp4"></video>
</div>
<div>
<p><b>B. Из атомов</b></p>
<img src="{rel_out}/still_атомы.jpg"/>
<video controls src="{rel_out}/B_из_атомов_видео_звук.mp4"></video>
</div>
</div>
<p><b>Рядом (без общего звука — слушай A/B выше)</b></p>
<video controls src="{rel_out}/рядом_живое_vs_атомы.mp4"></video>
<p><b>E:</b> B узнаётся как {style['имя'].lower()} (ухо+глаз)? да / почти / нет</p>
<p><a href="/выход/атомная_визуализация/E_все_живое_vs_атомы.html">← все стихии</a></p>
</body></html>
""",
        encoding="utf-8",
    )

    return {
        "id": style["id"],
        "имя": style["имя"],
        "t0": t0,
        "dur": dur,
        "режим": режим,
        "note": note or None,
        "E": "/" + style["e"],
        "A": "/" + str(Path(style["out"]) / "A_живое_видео_звук.mp4"),
        "B": "/" + str(Path(style["out"]) / "B_из_атомов_видео_звук.mp4"),
        "ok": True,
    }


def write_hub(results: list[dict]):
    cards = []
    for r in results:
        warn = f"<p style='color:#e0b070;font-size:.9rem'>⚠ {r['note']}</p>" if r.get("note") else ""
        cards.append(
            f"""<div class="card">
<h2>{r['имя']}</h2>
{warn}
<p class="meta">t0={r['t0']}с · {r['dur']}с · {r.get('режим') or '—'}</p>
<p><a class="btn" href="{r['E']}">Открыть сравнение</a></p>
</div>"""
        )
    HUB.parent.mkdir(parents=True, exist_ok=True)
    HUB.write_text(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — все: живое vs атомы</title>
<style>
body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:900px;margin:2rem auto;padding:0 1rem}}
.card{{border:1px solid #2a3340;border-radius:10px;padding:1rem 1.2rem;margin:0.8rem 0}}
.btn{{display:inline-block;margin-top:.4rem;padding:.45rem .9rem;background:#1e3a2f;color:#9dceb0;border-radius:6px;text-decoration:none}}
.meta{{opacity:.8;font-size:.92rem}} a{{color:#9dceb0}}
</style></head><body>
<h1>Живое видео+звук · vs · сборка из атомов</h1>
<p class="meta">{date.today().isoformat()} · по очереди все корпуса · слушай A и B раздельно</p>
{''.join(cards)}
<p><a href="/выход/атомная_визуализация/index.html">← хаб</a></p>
</body></html>
""",
        encoding="utf-8",
    )


def main() -> int:
    results = []
    for style in STYLES:
        try:
            results.append(build_one(style))
        except Exception as e:
            results.append({
                "id": style["id"], "имя": style["имя"], "ok": False, "error": str(e),
                "E": "/" + style["e"], "t0": None, "dur": None, "режим": None, "note": str(e),
                "A": None, "B": None,
            })
            print("FAIL", style["id"], e, flush=True)
    write_hub([r for r in results if r.get("ok")])
    # still list fails on hub with note
    if any(not r.get("ok") for r in results):
        # rewrite hub including fails
        write_hub(results) if False else None
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"дата": date.today().isoformat(), "results": results}, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # hub with all including errors
    cards = []
    for r in results:
        if r.get("ok"):
            warn = f"<p style='color:#e0b070;font-size:.9rem'>⚠ {r['note']}</p>" if r.get("note") else ""
            cards.append(
                f"<div class='card'><h2>{r['имя']}</h2>{warn}"
                f"<p class='meta'>t0={r['t0']}с · {r['dur']}с · {r.get('режим') or '—'}</p>"
                f"<p><a class='btn' href='{r['E']}'>Открыть сравнение</a></p></div>"
            )
        else:
            cards.append(
                f"<div class='card'><h2>{r['имя']}</h2>"
                f"<p style='color:#e08080'>FAIL: {r.get('error')}</p></div>"
            )
    HUB.write_text(
        f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/>
<title>E — все: живое vs атомы</title>
<style>
body{{font-family:system-ui;background:#0e1218;color:#e8eef6;max-width:900px;margin:2rem auto;padding:0 1rem}}
.card{{border:1px solid #2a3340;border-radius:10px;padding:1rem 1.2rem;margin:0.8rem 0}}
.btn{{display:inline-block;margin-top:.4rem;padding:.45rem .9rem;background:#1e3a2f;color:#9dceb0;border-radius:6px;text-decoration:none}}
.meta{{opacity:.8;font-size:.92rem}} a{{color:#9dceb0}}
</style></head><body>
<h1>Живое видео+звук · vs · сборка из атомов</h1>
<p class="meta">{date.today().isoformat()} · порядок: дождь → огонь → ветер → водопад → река → гром</p>
{''.join(cards)}
<p><a href="/выход/атомная_визуализация/index.html">← хаб</a></p>
</body></html>
""",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "n": len(results), "hub": str(HUB.relative_to(КОРЕНЬ))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

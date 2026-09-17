# -*- coding: utf-8 -*-
"""J1: собрать один атом причины дождя = геометрия + закон + окно звука."""
from __future__ import annotations

import json
import os
from datetime import date

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK = os.path.join(КОРЕНЬ, "отчёты", "трек_капли_дождь.json")
MODEL = os.path.join(КОРЕНЬ, "отчёты", "модель_капля_удар.json")
OUT = os.path.join(КОРЕНЬ, "выход", "атом_причины_дождь_001.json")
MD = os.path.join(КОРЕНЬ, "отчёты", "атом_причины_дождь_001.md")


def main() -> int:
    geo = json.load(open(TRACK, encoding="utf-8"))
    phys = json.load(open(MODEL, encoding="utf-8"))
    atom = {
        "id": "cause_dozhd_001",
        "тип": "атом_причины",
        "стихия": "дождь",
        "событие": "drop_impact",
        "дата": date.today().isoformat(),
        "правило_автора": "видео→геометрия; закон→причина звука; вместе→атом",
        "геометрия": {
            "video": geo["video"],
            "points": geo["points"],
            "t_impact": geo["t_impact"],
            "xy_impact": geo["xy_impact"],
            "v_px_per_s": geo.get("v_px_per_s"),
            "preview": geo.get("preview"),
            "точность": "приблизительная",
        },
        "физика": phys.get("модель"),
        "звук": {
            "synth_model_wav": phys.get("synth_wav"),
            "native_window_wav": phys.get("native_window_wav"),
            "band_corr_vs_native": phys.get("band_corr_vs_native"),
            "источник_оси": "native_aac video_live_01",
        },
        "образ_кусочек": {
            "смысл": "не png пейзажа, а модель капли+удар, калиброванная геометрией трека",
            "геометрия_намёк": geo.get("preview"),
        },
        "обратимость": {
            "звук_модель_есть": True,
            "сверка_с_native": phys.get("band_corr_vs_native"),
            "params_104": None,
            "долг_params_104": True,
            "note": "следующий долг — вписать 104+оси окна native вокруг t_impact",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(atom, f, ensure_ascii=False, indent=2)

    lines = [
        "# Атом причины дождь_001 (J1)",
        "",
        f"> t_impact={atom['геометрия']['t_impact']}s · band_corr={atom['звук']['band_corr_vs_native']}",
        "",
        f"- json: `{os.path.relpath(OUT, КОРЕНЬ)}`",
        f"- геометрия точек: {len(atom['геометрия']['points'])}",
        f"- r={atom['физика']['r_m']*1000:.2f} мм · v={atom['физика']['v_ms']:.2f} м/с",
        f"- synth: `{atom['звук']['synth_model_wav']}`",
        "",
        "Слушай модель и native window — вердикт E.",
        "",
    ]
    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps({"ok": True, "atom": OUT, "band_corr": atom["звук"]["band_corr_vs_native"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

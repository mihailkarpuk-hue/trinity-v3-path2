# -*- coding: utf-8 -*-
"""Вывод канонических 108 ключей из пересечения числовых полей 150 клеток каталога.

usage: python3 scripts/вывести_ключи_108.py [--write]
  --write  обновить ядро/ключи_108.py и данные/ключи_108.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, КОРЕНЬ)

КАТАЛОГ = os.path.join(КОРЕНЬ, "данные", "клеточки", "каталог.json")
ЗЕРКАЛО = os.path.join(КОРЕНЬ, "данные", "ключи_108.json")
МОДУЛЬ = os.path.join(КОРЕНЬ, "ядро", "ключи_108.py")

# Группы по ТЗ V3.3 (сумма = 108). Порядок ключей = порядок групп.
KEYS_BY_GROUP: dict[str, tuple[str, ...]] = {
    "TEMPORAL": (
        "amplitude_rms", "amplitude_peak", "amplitude_avg", "loudness_lufs",
        "crest_factor", "dynamic_range", "attack_time", "decay_time",
        "sustain_level", "release_time", "onset_density", "onset_strength",
        "bpm", "rhythm_stability", "bpm_isolation", "bpm_periodicity",
    ),
    "SPECTRAL": (
        "spectral_centroid", "spectral_spread", "spectral_skewness", "spectral_kurtosis",
        "spectral_flatness", "spectral_rolloff_85", "spectral_flux", "spectral_slope",
        "low_freq_ratio", "low_mid_freq_ratio", "mid_freq_ratio", "high_mid_freq_ratio",
        "high_freq_ratio", "pitch_fundamental", "harmonic_ratio", "inharmonicity",
        "noise_floor", "tonality",
    ),
    "VOCAL": (
        "formant_f1", "formant_f2", "formant_f3", "voicing_ratio",
        "breathiness", "roughness", "jitter", "shimmer",
    ),
    "MUSICAL": (
        "chroma_C", "chroma_Cs", "chroma_D", "chroma_Ds", "chroma_E", "chroma_F",
        "chroma_Fs", "chroma_G", "chroma_Gs", "chroma_A", "chroma_As", "chroma_B",
        "key_dominant", "consonance",
    ),
    "SPATIAL": (
        "stereo_width", "stereo_correlation", "spatial_impression",
        "reverb_estimate", "direct_to_reverb_ratio",
    ),
    "PERCEPTUAL": (
        "perceptual_sharpness", "perceptual_softness", "perceptual_tension",
        "perceptual_plushness", "perceptual_warmth", "perceptual_brightness",
        "perceptual_density",
    ),
    "MOVEMENT": (
        "movement_explosion", "movement_resonance", "movement_flow", "movement_mist",
        "movement_fracture", "self_similarity_index", "amplitude_modulation_smoothness",
        "ring_decay",
    ),
    "ADVANCED": tuple(
        [f"mfcc_{i}" for i in range(13)]
        + [f"mfcc_delta_{i}" for i in range(13)]
        + ["zero_crossing_rate"]
    ),
    "AXES": ("fd", "selfsim_r2", "nestedness", "mod_depth", "mod_rate"),
}


def пересечение_из_каталога() -> set[str]:
    """Числовые ключи params_104, общие для всех 150 клеток."""
    d = json.load(open(КАТАЛОГ, encoding="utf-8"))
    клетки = next(v for v in d.values() if isinstance(v, list))
    sets = []
    for c in клетки:
        p = c.get("параметры104") or {}
        sets.append({k for k, v in p.items() if isinstance(v, (int, float))})
    return set.intersection(*sets) if sets else set()


def проверить_канон() -> list[str]:
    keys = [k for g in KEYS_BY_GROUP.values() for k in g]
    assert len(keys) == 108, f"ожидалось 108, есть {len(keys)}"
    assert len(set(keys)) == 108, "дубликаты в KEYS_BY_GROUP"
    for name, g in KEYS_BY_GROUP.items():
        assert len(g) == {"TEMPORAL": 16, "SPECTRAL": 18, "VOCAL": 8, "MUSICAL": 14,
                          "SPATIAL": 5, "PERCEPTUAL": 7, "MOVEMENT": 8, "ADVANCED": 27,
                          "AXES": 5}[name], f"неверный размер {name}"
    common = пересечение_из_каталога()
    missing = set(keys) - common
    extra = common - set(keys)
    if missing:
        raise SystemExit(f"ключи канона отсутствуют в каталоге: {sorted(missing)}")
    if extra:
        raise SystemExit(f"лишние ключи в каталоге (не в каноне): {sorted(extra)}")
    return keys


def sha256_keys(keys: list[str]) -> str:
    payload = "\n".join(keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def записать_модуль(keys: list[str], sha: str) -> None:
    lines = [
        "# -*- coding: utf-8 -*-",
        '"""Канон 108 ключей params_104 — единый порядок для метрики d₉ и индексов.',
        "",
        "Сгенерировано scripts/вывести_ключи_108.py — не редактировать вручную.",
        '"""',
        "from __future__ import annotations",
        "",
        "KEYS_BY_GROUP: dict[str, tuple[str, ...]] = {",
    ]
    for name, group in KEYS_BY_GROUP.items():
        items = ", ".join(f'"{k}"' for k in group)
        lines.append(f'    "{name}": ({items}),')
    lines.append("}")
    lines.append("")
    lines.append("KEYS_108: list[str] = [k for g in KEYS_BY_GROUP.values() for k in g]")
    lines.append(f'KEYS_SHA256: str = "{sha}"')
    lines.append("")
    with open(МОДУЛЬ, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def записать_зеркало(keys: list[str], sha: str) -> None:
    doc = {
        "версия": 1,
        "ключи": keys,
        "группы": {g: list(k) for g, k in KEYS_BY_GROUP.items()},
        "sha256": sha,
    }
    with open(ЗЕРКАЛО, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    keys = проверить_канон()
    sha = sha256_keys(keys)
    print(f"✓ 108 ключей · sha256={sha[:16]}…")
    print(f"  клеток в каталоге: 150 · пересечение совпадает с каноном")
    if "--write" in sys.argv:
        записать_модуль(keys, sha)
        записать_зеркало(keys, sha)
        print(f"✓ записано: {os.path.relpath(МОДУЛЬ, КОРЕНЬ)}")
        print(f"✓ записано: {os.path.relpath(ЗЕРКАЛО, КОРЕНЬ)}")
    else:
        # также сохранить в /tmp для совместимости с построить_индекс_108.py
        tmp = os.path.join("/tmp", "keys108.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(keys, f, ensure_ascii=False)
        print(f"✓ /tmp/keys108.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Per-atom params_104: окно ±50 ms вокруг birth → analyze_full_103 + 5 осей.

Использование:
  python3 обогатить_атомы_104.py буквица_живая/живая_У_протяжное.json
  python3 обогатить_атомы_104.py --pilot   # У протяжное + etalon дождь
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
ПРОЕКТ = os.path.dirname(КОРЕНЬ)
sys.path.insert(0, os.path.join(ПРОЕКТ, "scripts"))
sys.path.insert(0, os.path.join(КОРЕНЬ, "ядро"))

from atoms_full103 import analyze_full_103  # noqa: E402
from оси import оси_звука  # noqa: E402
from кресты import собрать_клетку  # noqa: E402
from фаза import загрузить  # noqa: E402

from ядро.пороги import WINDOW_MS_MAX, WINDOW_MS_MIN, SR as SR_ATOM_DEFAULT

WINDOW_MS = WINDOW_MS_MAX  # legacy alias
SR_ATOM = 16000


def _окно_ms(atom: dict) -> float:
    """Адаптивное окно: clip(lifetime, 50мс, 200мс)."""
    lt = float(atom.get("lifetime") or 0.05)
    return float(max(WINDOW_MS_MIN, min(WINDOW_MS_MAX, lt * 1000.0)))


def _окно_атома(x: np.ndarray, sr: int, atom: dict) -> np.ndarray:
    half = int(sr * _окно_ms(atom) / 2000.0)
    center = int(float(atom.get("birth") or 0) * sr + float(atom.get("lifetime") or 0) * sr * 0.5)
    i0 = max(0, center - half)
    i1 = min(len(x), center + half)
    chunk = x[i0:i1]
    min_len = max(sr // 50, 2048)
    if len(chunk) < min_len:
        chunk = np.pad(chunk, (0, min_len - len(chunk)))
    return chunk


def обогатить_оси_из_звука(
    atoms: list[dict],
    x: np.ndarray,
    sr: int,
    *,
    target_sr: int = SR_ATOM,
) -> None:
    """Быстрое обогащение params_104 осями (fd, nestedness, …) из массива — для round-trip."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m
    if sr != target_sr:
        from scipy.signal import resample
        x = resample(x, int(len(x) * target_sr / sr)).astype(np.float64)
        sr = target_sr
    for atom in atoms:
        birth = float(atom.get("birth") or 0)
        chunk = _окно_атома(x, sr, atom)
        оси = оси_звука(chunk, sr)
        p = atom.get("params_104") or {}
        atom["params_104"] = {**p, **оси}


def обогатить_json(json_path: str, audio_path: str | None = None, *, in_place: bool = True, force: bool = False) -> dict:
    with open(json_path, encoding="utf-8") as f:
        rec = json.load(f)
    atoms = rec.get("atoms") or []
    if not atoms:
        raise ValueError(f"нет atoms: {json_path}")

    if audio_path is None:
        base = os.path.splitext(json_path)[0]
        for ext in (".m4a", ".wav", ".mp3"):
            p = base + ext
            if os.path.isfile(p):
                audio_path = p
                break
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"звук не найден для {json_path}")

    x, sr = загрузить(audio_path) if audio_path.endswith(".wav") else _load_any(audio_path)
    if sr != SR_ATOM:
        from scipy.signal import resample

        n = int(len(x) * SR_ATOM / sr)
        x = resample(x, n).astype(np.float64)
        sr = SR_ATOM

    t0 = time.time()
    for i, atom in enumerate(atoms):
        if not force and atom.get("params_104") and len(atom["params_104"]) >= 100:
            continue
        birth = float(atom.get("birth") or 0)
        chunk = _окно_атома(x, sr, atom)
        p103 = analyze_full_103(chunk, sr)
        оси = оси_звука(chunk, sr)
        atom["params_104"] = {**p103, **оси}

    cell = собрать_клетку(atoms, meta={
        "version": rec.get("version", "3.0-atom-first"),
        "длительность_сек": rec.get("длительность_сек") or rec.get("duration"),
    })
    rec["atoms"] = cell["atoms"]
    rec["crosses"] = cell["crosses"]
    rec["число_связей"] = cell["число_связей"]
    rec["atoms_count"] = len(atoms)
    rec["version"] = cell.get("version") or "3.0-atom-first"

    # агрегат клетки — среднее по атомам (производное)
    keys = [k for k in (atoms[0].get("params_104") or {}) if isinstance(atoms[0]["params_104"][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [float(a["params_104"][k]) for a in atoms if k in a.get("params_104", {})]
        if vals:
            agg[k] = round(float(np.mean(vals)), 6)
    rec["параметры104"] = agg
    rec["parent_params_full"] = agg  # совместимость со старым UI

    elapsed = time.time() - t0
    if in_place:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
    print(f"✓ {os.path.basename(json_path)}: {len(atoms)} атомов · {rec['число_связей']} крестов · 104/атом · {elapsed:.1f}s")
    return rec


def _load_any(path: str):
    """m4a/mp3 → mono float 16 kHz."""
    import shutil
    import subprocess
    import tempfile

    from scipy.io import wavfile

    f = tempfile.mktemp(suffix=".wav")
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SR_ATOM), "-sample_fmt", "s16", f],
            capture_output=True,
        )
    elif shutil.which("afconvert"):
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR_ATOM}", "-c", "1", path, f],
            capture_output=True,
            check=True,
        )
    else:
        raise RuntimeError("нужен ffmpeg или afconvert для m4a/mp3")
    sr, x = wavfile.read(f)
    os.remove(f)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m
    return x, int(sr)


def _обновить_каталог(sid: str, rec: dict) -> None:
    cat_path = os.path.join(КОРЕНЬ, "данные", "клеточки", "каталог.json")
    with open(cat_path, encoding="utf-8") as f:
        d = json.load(f)
    key = next(k for k, v in d.items() if isinstance(v, list))
    for c in d[key]:
        if c.get("id") == sid:
            c["число_связей"] = rec.get("число_связей", 0)
            if rec.get("параметры104"):
                c["параметры104"] = rec["параметры104"]
            break
    with open(cat_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def _all_bukvitsa(force: bool = False):
    import glob

    kl = os.path.join(КОРЕНЬ, "данные", "клеточки", "буквица_живая")
    paths = sorted(glob.glob(os.path.join(kl, "живая_*.json")))
    t0 = time.time()
    ok = 0
    for i, p in enumerate(paths, 1):
        sid = os.path.basename(p).replace(".json", "")
        try:
            rec = обогатить_json(p, force=force)
            _обновить_каталог(sid, rec)
            ok += 1
        except Exception as e:
            print(f"✗ {sid}: {e}")
    print(f"ГОТОВО: {ok}/{len(paths)} за {time.time()-t0:.0f}s · каталог обновлён")


def _pilot():
    kl = os.path.join(КОРЕНЬ, "данные", "клеточки")
    paths = [
        os.path.join(kl, "буквица_живая", "живая_У_протяжное.json"),
        os.path.join(kl, "клеточки_полные", "etalon_dozhd.json"),
    ]
    for p in paths:
        if os.path.isfile(p):
            обогатить_json(p)


if __name__ == "__main__":
    force = "--force" in sys.argv
    if len(sys.argv) > 1 and sys.argv[1] == "--all-bukvitsa":
        _all_bukvitsa(force=force)
    elif len(sys.argv) > 1 and sys.argv[1] == "--pilot":
        _pilot()
    elif len(sys.argv) > 1 and sys.argv[1] not in ("--force",):
        jp = sys.argv[1]
        if not os.path.isabs(jp):
            cand = os.path.join(КОРЕНЬ, jp)
            jp = cand if os.path.isfile(cand) else os.path.join(КОРЕНЬ, "данные", "клеточки", jp)
        обогатить_json(jp, force=force)
    else:
        print("usage: python3 обогатить_атомы_104.py <json> | --pilot | --all-bukvitsa [--force]")

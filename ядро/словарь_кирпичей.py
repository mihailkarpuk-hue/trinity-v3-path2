# -*- coding: utf-8
"""Словарь первокирпичей — типы атомов как распределения в пространстве ГЕН.

Тип ≠ место: вектор типа БЕЗ birth, freq, amp, freq_t, amp_t.
Кирпич = центроид + σ по осям + доля корпуса + имя (не источник).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Sequence

import numpy as np

from ядро.ген import ген_из_атома, валиден

КОРЕНЬ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
КЛЕТКИ = os.path.join(КОРЕНЬ, "данные", "клеточки")
КАТАЛОГ = os.path.join(КЛЕТКИ, "каталог.json")
СЛОВАРЬ_ПУТЬ = os.path.join(КОРЕНЬ, "данные", "словарь_кирпичей.json")
ОСТАТОК_ПУТЬ = os.path.join(КОРЕНЬ, "данные", "кирпичи_остаток.json")

# 18 базис-эталонов + 37 живых букв (55 обогащённых клеток)
КОРПУС_18_БАЗИС = (
    "bazis_02_pila_etalon", "bazis_07_am_etalon", "bazis_08_fm_etalon",
    "bazis_11_malyi_baraban", "bazis_12_kaplya",
    "etalon_dozhd", "etalon_ogon", "etalon_veter", "etalon_vodopad",
    "etalon_grom", "etalon_pesok", "etalon_serdce", "etalon_dyhanie",
    "etalon_komar", "etalon_hrap", "etalon_kashel", "etalon_chihanie",
    "etalon_lyagushki",
)

ТИП_КЛЮЧИ = (
    "фаза_тон", "фаза_шум", "фаза_переход",
    "log_lifetime", "freq_slope", "harmonicity", "attack_ratio",
    "decay_shape", "noise_ratio", "band_width_rel", "mod_depth", "mod_rate",
)

K_MIN, K_MAX = 12, 48
SEED = 42
NULL_MARGIN = 0.1


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def вектор_типа_из_гена(ген: dict) -> np.ndarray:
    """ГЕН → вектор типа (12 осей, без места)."""
    ф = str(ген.get("фаза") or "тон")
    lt = max(1e-4, float(ген.get("lifetime") or 0.05))
    freq = max(1.0, float(ген.get("freq") or 440))
    bw = float(ген.get("band_width") or freq * 0.25)
    return np.array([
        1.0 if ф == "тон" else 0.0,
        1.0 if ф == "шум" else 0.0,
        1.0 if ф == "переход" else 0.0,
        math.log(lt),
        float(ген.get("freq_slope") or 0),
        float(ген.get("harmonicity") or 0),
        float(ген.get("attack_ratio") or 0.1),
        float(ген.get("decay_shape") or -2.0),
        float(ген.get("noise_ratio") or 0),
        bw / freq,
        float(ген.get("mod_depth") or 0),
        float(ген.get("mod_rate") or 0),
    ], dtype=np.float64)


def _фаза_из_вектора(v: np.ndarray) -> str:
    i = int(np.argmax(v[:3]))
    return ("тон", "шум", "переход")[i]


def _траектории_гена(g: dict, n: int = 8) -> tuple[list[float], list[float]]:
    """Синтетические freq_t/amp_t из полей типа (если нет в скелете)."""
    lt = max(1e-3, float(g.get("lifetime") or 0.05))
    t = np.linspace(0, lt, max(2, min(n, 32)))
    f0 = float(g.get("freq") or 440)
    slope = float(g.get("freq_slope") or 0)
    freqs = (f0 * (2.0 ** (slope * t))).tolist()
    ar = float(g.get("attack_ratio") or 0.1)
    ds = float(g.get("decay_shape") or -2.0)
    amps = np.ones(len(t), dtype=np.float64)
    i_peak = max(1, min(len(t) - 1, int(ar * len(t))))
    amps[:i_peak] = np.linspace(0.05, 1.0, i_peak)
    tail = len(t) - i_peak
    if tail > 0:
        tt = np.arange(tail, dtype=np.float64) / max(lt, 1e-3)
        amps[i_peak:] = np.exp(ds * tt * lt) if ds < 0 else np.linspace(1.0, 0.1, tail)
    return freqs, amps.tolist()


def ген_из_вектора_типа(
    v: np.ndarray,
    *,
    birth: float,
    freq: float,
    amp: float,
    freq_t: list | None = None,
    amp_t: list | None = None,
) -> dict:
    """Денормализованный вектор типа + скелет → полный ГЕН."""
    lt = float(np.exp(v[3]))
    bw_rel = max(0.05, float(v[9]))
    ф = _фаза_из_вектора(v)
    h = float(np.clip(v[5], 0, 1))
    g = {
        "фаза": ф,
        "birth": float(birth),
        "lifetime": lt,
        "freq": float(freq),
        "freq_slope": float(v[4]),
        "harmonic_index": 1 if h > 0.5 else 0,
        "harmonicity": h,
        "amp": float(amp),
        "attack_ratio": float(np.clip(v[6], 0, 1)),
        "decay_shape": float(v[7]),
        "noise_ratio": float(np.clip(v[8], 0, 1)),
        "band_center": float(freq),
        "band_width": float(bw_rel * freq),
        "mod_depth": float(max(0, v[10])),
        "mod_rate": float(max(0, v[11])),
        "freq_t": list(freq_t or []),
        "amp_t": list(amp_t or []),
    }
    if not g["freq_t"] or not g["amp_t"]:
        ft, at = _траектории_гена(g)
        g["freq_t"] = ft
        g["amp_t"] = at
    return g


def zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, mu, sd


def перемешать_оси(X: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Нулевой тест: маргинали сохранены, связи разрушены."""
    rng = _rng(seed)
    Y = X.copy()
    for j in range(X.shape[1]):
        Y[:, j] = rng.permutation(X[:, j])
    return Y


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.dot(d, d)))


def _kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(X)
    centroids = np.empty((k, X.shape[1]), dtype=np.float64)
    centroids[0] = X[int(rng.integers(n))]
    d2 = np.sum((X - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centroids[i] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centroids[i]) ** 2, axis=1))
    return centroids


def k_means(
    X: np.ndarray,
    k: int,
    *,
    seed: int = SEED,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """k-means, детерминированный seed."""
    rng = _rng(seed)
    centroids = _kmeans_pp(X, k, rng)
    labels = np.zeros(len(X), dtype=np.int32)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
    return labels, centroids


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Средний silhouette по точкам (евклид)."""
    n = len(X)
    k = int(labels.max()) + 1
    if k < 2 or n < k + 1:
        return 0.0
    scores = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = float(np.mean([_dist(X[i], X[j]) for j in np.where(same)[0]]))
        b = math.inf
        for c in range(k):
            if c == labels[i]:
                continue
            other = labels == c
            if not other.any():
                continue
            b = min(b, float(np.mean([_dist(X[i], X[j]) for j in np.where(other)[0]])))
        if math.isinf(b):
            continue
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores)) if scores else 0.0


def кривая_silhouette(X: np.ndarray, k_range: range | None = None) -> dict[int, float]:
    k_range = k_range or range(K_MIN, K_MAX + 1)
    out = {}
    for k in k_range:
        if k >= len(X):
            break
        labels, _ = k_means(X, k, seed=SEED)
        out[k] = round(silhouette(X, labels), 6)
    return out


def нулевой_тест(X: np.ndarray) -> dict[str, Any]:
    """Silhouette на реальных vs перемешанных осях."""
    X_shuf = перемешать_оси(X, SEED)
    real = кривая_silhouette(X)
    shuf = кривая_silhouette(X_shuf)
    лучший_k = None
    лучший_запас = -math.inf
    for k in real:
        if k not in shuf:
            continue
        margin = real[k] - shuf[k]
        if margin > лучший_запас:
            лучший_запас = margin
            лучший_k = k
    пройден = лучший_запас >= NULL_MARGIN
    return {
        "пройден": пройден,
        "лучший_k": лучший_k,
        "лучший_запас": round(лучший_запас, 6),
        "порог": NULL_MARGIN,
        "silhouette_реальные": real,
        "silhouette_шум": shuf,
    }


def _имя_типа(c: dict[str, float]) -> str:
    """Автоимя по доминирующим полям (не по источнику)."""
    parts = []
    if c.get("attack_ratio", 0) > 0.35 and c.get("log_lifetime", 0) < math.log(0.08):
        parts.append("удар-звон")
    elif c.get("noise_ratio", 0) > 0.55:
        parts.append("шипящее зерно" if c.get("band_width_rel", 0) > 0.4 else "шорох")
    elif c.get("harmonicity", 0) > 0.65:
        parts.append("тональная нить")
    if c.get("freq_slope", 0) > 1.5:
        parts.append("чирп-вверх")
    elif c.get("freq_slope", 0) < -1.5:
        parts.append("чирп-вниз")
    if c.get("mod_depth", 0) > 0.15:
        parts.append("пульс")
    if c.get("фаза_шум", 0) > 0.5 and not parts:
        parts.append("шумовое зерно")
    if c.get("фаза_переход", 0) > 0.5 and not parts:
        parts.append("переход")
    if not parts:
        parts.append("смешанный")
    return "-".join(parts[:2])


def _centroid_dict(c: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict[str, float]:
    raw = c * sd + mu
    d = {k: round(float(raw[i]), 6) for i, k in enumerate(ТИП_КЛЮЧИ)}
    d["имя"] = _имя_типа(d)
    return d


def типы_из_кластеров(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    метаданные: list[dict],
) -> list[dict]:
    """Кластеры → типы-распределения."""
    k = len(centroids)
    n = len(X)
    типы = []
    for j in range(k):
        mask = labels == j
        if not mask.any():
            continue
        pts = X[mask]
        sigma = pts.std(axis=0)
        sigma[sigma < 1e-6] = sd[sigma < 1e-6] if np.any(sigma < 1e-6) else 1e-3
        dists = np.linalg.norm(pts - centroids[j], axis=1)
        d95 = float(np.percentile(dists, 95)) if len(dists) > 1 else 0.5
        источники = sorted({m["клетка"] for m, keep in zip(метаданные, mask) if keep})
        c_dict = _centroid_dict(centroids[j], mu, sd)
        типы.append({
            "id": j,
            "имя": c_dict.pop("имя"),
            "доля": round(float(mask.sum()) / n, 6),
            "центроид": c_dict,
            "sigma": {k2: round(float(sigma[i] * sd[i]), 6) for i, k2 in enumerate(ТИП_КЛЮЧИ)},
            "d95": round(d95, 6),
            "происхождение": источники,
        })
    return типы


def _seed_atom(birth: float, freq: float, typ_id: int) -> int:
    payload = f"{birth}:{freq}:{typ_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def сэмпл_вектора_типа(
    тип: dict,
    birth: float,
    freq: float,
    mu: np.ndarray,
    sd: np.ndarray,
    *,
    живость: float = 0.0,
) -> np.ndarray:
    """N(центроид, σ·живость) в z-пространстве, детерминированно."""
    rng = _rng(_seed_atom(birth, freq, int(тип["id"])))
    c = np.array([тип["центроид"][k] for k in ТИП_КЛЮЧИ], dtype=np.float64)
    sig = np.array([тип["sigma"][k] for k in ТИП_КЛЮЧИ], dtype=np.float64)
    raw = c + rng.normal(size=len(c)) * sig * float(живость)
    return (raw - mu) / sd


def _вектор_типа_кирпича(
    тип: dict,
    atom: dict,
    mu: np.ndarray,
    sd: np.ndarray,
    cent_z: np.ndarray | None,
    *,
    живость: float = 0.0,
    blend: bool = True,
    ton_cap: float | None = None,
) -> np.ndarray:
    """Тип кирпича: blend экземпляра к центроиду по d95."""
    g0 = ген_из_атома(atom)
    v_atom = вектор_типа_из_гена(g0)
    v_cent = np.array([тип["центроид"][k] for k in ТИП_КЛЮЧИ], dtype=np.float64)
    if not blend:
        return v_atom
    if cent_z is None:
        vz = сэмпл_вектора_типа(тип, g0["birth"], g0["freq"], mu, sd, живость=живость)
        return vz * sd + mu
    vz_atom = (v_atom - mu) / sd
    tid = int(тип["id"])
    d = float(np.linalg.norm(vz_atom - cent_z[tid]))
    w = min(1.0, d / (float(тип.get("d95") or 0.5) + 1e-9))
    if ton_cap is not None and float(g0.get("harmonicity") or 0) > 0.55:
        w = min(w, float(ton_cap))
    rng = _rng(_seed_atom(g0["birth"], g0["freq"], tid))
    sig = np.array([тип["sigma"][k] for k in ТИП_КЛЮЧИ], dtype=np.float64)
    v_jitter = v_cent + rng.normal(size=len(v_cent)) * sig * float(живость)
    return (1.0 - w) * v_atom + w * v_jitter


def сэмпл_гена(
    тип: dict,
    atom: dict,
    mu: np.ndarray,
    sd: np.ndarray,
    *,
    cent_z: np.ndarray | None = None,
    живость: float = 0.0,
    blend: bool = True,
    ton_cap: float | None = None,
) -> dict:
    """Скелет атома (место + траектории) + плоть из распределения типа."""
    g0 = ген_из_атома(atom)
    v = _вектор_типа_кирпича(
        тип, atom, mu, sd, cent_z, живость=живость, blend=blend, ton_cap=ton_cap,
    )
    g = ген_из_вектора_типа(
        v,
        birth=g0["birth"],
        freq=g0["freq"],
        amp=g0["amp"],
        freq_t=g0.get("freq_t") or None,
        amp_t=g0.get("amp_t") or None,
    )
    if g0.get("режим"):
        g["режим"] = g0["режим"]
    return g


def ближайший_тип(vz: np.ndarray, centroids: np.ndarray) -> int:
    dists = np.linalg.norm(centroids - vz, axis=1)
    return int(np.argmin(dists))


def _cells_catalog() -> list[dict]:
    d = json.load(open(КАТАЛОГ, encoding="utf-8"))
    return next(v for v in d.values() if isinstance(v, list))


def id_корпуса() -> list[str]:
    ids = list(КОРПУС_18_БАЗИС)
    for c in _cells_catalog():
        if c.get("группа") == "буквица_живая":
            ids.append(c["id"])
    return ids


def загрузить_атомы_корпуса() -> tuple[np.ndarray, list[dict], list[dict]]:
    """Все атомы корпуса → (X_norm, mu, sd), метаданные, исходные атомы."""
    ids = id_корпуса()
    by_id = {c["id"]: c for c in _cells_catalog()}
    vecs = []
    meta = []
    atoms_raw = []
    for cid in ids:
        c = by_id.get(cid)
        if not c:
            continue
        jp = os.path.join(КЛЕТКИ, c.get("атомы") or "")
        if not os.path.isfile(jp):
            continue
        rec = json.load(open(jp, encoding="utf-8"))
        for i, atom in enumerate(rec.get("atoms") or []):
            g = ген_из_атома(atom)
            if not валиден(g):
                continue
            vecs.append(вектор_типа_из_гена(g))
            meta.append({"клетка": cid, "индекс": i})
            atoms_raw.append(atom)
    X = np.asarray(vecs, dtype=np.float64)
    Xn, mu, sd = zscore(X)
    return Xn, mu, sd, meta, atoms_raw


def построить_словарь() -> dict[str, Any]:
    """Полный пайплайн: нулевой тест → кластеризация → типы."""
    Xn, mu, sd, meta, _ = загрузить_атомы_корпуса()
    null = нулевой_тест(Xn)
    путь = "kmeans" if null["пройден"] else "kmedoids_soft"

    if null["пройден"]:
        k = null["лучший_k"] or K_MIN
        labels, centroids = k_means(Xn, k, seed=SEED)
        sil = silhouette(Xn, labels)
        типы = типы_из_кластеров(Xn, labels, centroids, mu, sd, meta)
    else:
        k = K_MIN
        labels, centroids = k_means(Xn, k, seed=SEED)
        sil = silhouette(Xn, labels)
        типы = типы_из_кластеров(Xn, labels, centroids, mu, sd, meta)
        путь = "kmedoids_soft"

    # остаток: d > d95 своего кластера ИЛИ d до всех centroids > глобальный порог
    остаток = []
    for i, (vz, m) in enumerate(zip(Xn, meta)):
        tid = int(labels[i])
        d_own = float(np.linalg.norm(vz - centroids[tid]))
        d_all = float(np.min(np.linalg.norm(centroids - vz, axis=1)))
        тип = next(t for t in типы if t["id"] == tid)
        if d_own > тип["d95"] * 1.05 or d_all > тип["d95"] * 1.5:
            остаток.append({**m, "d_кластер": round(d_own, 4), "тип": tid})

    словарь = {
        "версия": 1,
        "seed": SEED,
        "k": int(k),
        "путь": путь,
        "корпус": id_корпуса(),
        "число_атомов": len(Xn),
        "тип_ключи": list(ТИП_КЛЮЧИ),
        "mu": [round(float(x), 6) for x in mu],
        "sd": [round(float(x), 6) for x in sd],
        "нулевой_тест": null,
        "silhouette": round(sil, 6),
        "типы": типы,
        "centroids_z": [[round(float(x), 6) for x in row] for row in centroids],
    }
    словарь["sha256"] = hashlib.sha256(
        json.dumps(словарь, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return словарь, labels, Xn, mu, sd, meta, остаток


def сохранить(словарь: dict, остаток: list) -> None:
    with open(СЛОВАРЬ_ПУТЬ, "w", encoding="utf-8") as f:
        json.dump(словарь, f, ensure_ascii=False, indent=1)
    with open(ОСТАТОК_ПУТЬ, "w", encoding="utf-8") as f:
        json.dump({"версия": 1, "атомов": len(остаток), "остаток": остаток}, f, ensure_ascii=False, indent=1)


def загрузить() -> dict:
    return json.load(open(СЛОВАРЬ_ПУТЬ, encoding="utf-8"))


def _cell(cid: str) -> dict:
    for c in _cells_catalog():
        if c["id"] == cid:
            return c
    raise KeyError(cid)



def собрать_гены_из_атомов(
    atoms: list,
    словарь: dict,
    *,
    labels_map: dict[tuple[str, int], int] | None = None,
    cid: str | None = None,
    живость: float = 0.0,
    blend: bool = True,
) -> tuple[list[dict], float, dict[str, float]]:
    """Список атомов → гены через кирpичи (blend)."""
    типы = {int(t["id"]): t for t in словарь["типы"]}
    mu = np.asarray(словарь["mu"], dtype=np.float64)
    sd = np.asarray(словарь["sd"], dtype=np.float64)
    cent = np.asarray(словарь["centroids_z"], dtype=np.float64)
    gens: list[dict] = []
    counts: dict[str, int] = {}
    for i, atom in enumerate(atoms):
        g0 = ген_из_атома(atom)
        if labels_map and cid and (cid, i) in labels_map:
            tid = int(labels_map[(cid, i)])
        else:
            vz = (вектор_типа_из_гена(g0) - mu) / sd
            tid = ближайший_тип(vz, cent)
        typ = типы[tid]
        tc = 0.30 if cid and _cell(cid).get("группа") == "буквица_живая" else None
        gens.append(сэмпл_гена(typ, atom, mu, sd, cent_z=cent, живость=живость, blend=blend, ton_cap=tc if blend else None))
        name = str(typ["имя"])
        counts[name] = counts.get(name, 0) + 1
    total = len(gens) or 1
    состав = {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda x: -x[1])}
    dur = max((g["birth"] + g["lifetime"] for g in gens), default=0.1) + 0.05
    return gens, dur, состав


def собрать_гены_клетки_адаптивно(
    cid: str,
    словарь: dict | None = None,
    *,
    labels_map: dict[tuple[str, int], int] | None = None,
    audio_x=None,
    audio_sr: int = 22050,
    живость: float = 0.0,
) -> tuple[list[dict], float, dict[str, float], str]:
    """Клетка → гены; для буквицы пробует stored и свежую атомизацию."""
    словарь = словарь or загрузить()
    c = _cell(cid)
    jp = os.path.join(КЛЕТКИ, c.get("атомы") or "")
    stored = json.load(open(jp, encoding="utf-8")).get("atoms") or []
    gens_s, dur_s, comp_s = собрать_гены_из_атомов(
        stored, словарь, labels_map=labels_map, cid=cid, живость=живость,
    )
    if c.get("группа") != "буквица_живая" or audio_x is None:
        return gens_s, dur_s, comp_s, "stored"
    from ядро.атомизация import атомизировать
    fresh = [{k: v for k, v in a.items() if not k.startswith("_")} for a in атомизировать(audio_x, audio_sr)]
    if len(fresh) <= len(stored):
        return gens_s, dur_s, comp_s, "stored"
    gens_f, dur_f, comp_f = собрать_гены_из_атомов(fresh, словарь, живость=живость)
    return gens_f, dur_f, comp_f, "fresh"


def собрать_гены_клетки(
    cid: str,
    словарь: dict | None = None,
    *,
    labels_map: dict[tuple[str, int], int] | None = None,
    живость: float = 0.0,
    blend: bool = True,
) -> tuple[list[dict], float, dict[str, float]]:
    """Атомы клетки → гены из кирпичей. Место атома сохраняется, тип — из словаря."""
    словарь = словарь or загрузить()
    c = _cell(cid)
    jp = os.path.join(КЛЕТКИ, c.get("атомы") or "")
    типы = {int(t["id"]): t for t in словарь["типы"]}
    mu = np.asarray(словарь["mu"], dtype=np.float64)
    sd = np.asarray(словарь["sd"], dtype=np.float64)
    cent = np.asarray(словарь["centroids_z"], dtype=np.float64)
    atoms = json.load(open(jp, encoding="utf-8")).get("atoms") or []
    gens: list[dict] = []
    counts: dict[str, int] = {}
    for i, atom in enumerate(atoms):
        g0 = ген_из_атома(atom)
        if labels_map and (cid, i) in labels_map:
            tid = int(labels_map[(cid, i)])
        else:
            vz = (вектор_типа_из_гена(g0) - mu) / sd
            tid = ближайший_тип(vz, cent)
        typ = типы[tid]
        tc = 0.30 if cid and _cell(cid).get("группа") == "буквица_живая" else None
        gens.append(сэмпл_гена(typ, atom, mu, sd, cent_z=cent, живость=живость, blend=blend, ton_cap=tc if blend else None))
        name = str(typ["имя"])
        counts[name] = counts.get(name, 0) + 1
    total = len(gens) or 1
    состав = {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda x: -x[1])}
    dur = max((g["birth"] + g["lifetime"] for g in gens), default=0.1) + 0.05
    return gens, dur, состав

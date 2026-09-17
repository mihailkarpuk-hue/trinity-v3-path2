# -*- coding: utf-8 -*-
"""нормировка_калибровки.py — шов 3: воспроизвести n_хвост/n_тепло/n_шероховатость
из полного 104-паспорта (мульти-линейная подгонка), записать в параметры104.

До r>0.95 по каждому полю — п104 в каталоге не удаляем (защита картинки).
"""
import os
import json
import shutil
import numpy as np

КОРЕНЬ = os.path.dirname(os.path.abspath(__file__))
кат = os.path.join(КОРЕНЬ, "данные", "клеточки", "каталог.json")
формула_файл = os.path.join(КОРЕНЬ, "данные", "формула_калибровки.json")

цели = ["n_хвост", "n_тепло", "n_шероховатость"]
TOP_K = 5  # сколько параметров 104 в мульти-регрессии


def col(items, src, k):
    return np.array([float(it[src].get(k, 0) or 0) for it in items])


def подогнать_multi(x_mat, y):
    """OLS: y ≈ X @ w, intercept в последнем столбце."""
    n = x_mat.shape[0]
    X = np.column_stack([x_mat, np.ones(n)])
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ w
    pred = np.clip(pred, 0, 1)
    mae = float(np.mean(np.abs(pred - y)))
    r = float(np.corrcoef(pred, y)[0, 1]) if y.std() > 1e-9 else 0.0
    return w, pred, mae, r


def main():
    # бэкап перед записью
    bak = кат + ".bak_шов3"
    if not os.path.exists(bak):
        shutil.copy2(кат, bak)
        print(f"бэкап: {bak}")

    d = json.load(open(кат, encoding="utf-8"))
    items = d if isinstance(d, list) else next((v for v in d.values() if isinstance(v, list)), [])
    есть = [it for it in items if "параметры104" in it and "п104" in it]
    if not есть:
        print("нет клеток с параметры104 и п104"); return

    ключи104 = [
        k for k in есть[0]["параметры104"].keys()
        if isinstance(есть[0]["параметры104"][k], (int, float))
        and not k.startswith("n_")
    ]
    print(f"клеток: {len(есть)} · кандидатов 104: {len(ключи104)}")

    формула = {"версия": 2, "метод": "multi_linear_top5", "поля": {}}

    for цель in цели:
        y = np.array([float(it["п104"].get(цель, 0) or 0) for it in есть])
        # ранжируем одиночные корреляции
        ранги = []
        for k in ключи104:
            x = col(есть, "параметры104", k)
            if x.std() < 1e-9:
                continue
            r = abs(float(np.corrcoef(x, y)[0, 1]))
            ранги.append((k, r))
        ранги.sort(key=lambda t: t[1], reverse=True)
        топ = [k for k, _ in ранги[:TOP_K]]
        if not топ:
            print(f"  {цель}: нет кандидатов"); continue

        x_mat = np.column_stack([col(есть, "параметры104", k) for k in топ])
        w, pred, mae, r = подогнать_multi(x_mat, y)

        # одиночная (старый метод) для сравнения
        k1 = топ[0]
        a1, b1 = np.polyfit(col(есть, "параметры104", k1), y, 1)
        pred1 = np.clip(a1 * col(есть, "параметры104", k1) + b1, 0, 1)
        r1 = float(np.corrcoef(pred1, y)[0, 1])

        формула["поля"][цель] = {
            "из": топ,
            "веса": [float(w[i]) for i in range(len(топ))],
            "b": float(w[-1]),
            "воспр_MAE": mae,
            "воспр_r": r,
            "single_r": r1,
        }
        print(f"  {цель:16s} multi r={r:.3f} MAE={mae:.3f}  (single r={r1:.3f})  ← {', '.join(топ[:3])}…")

    # записать n_* в параметры104
    for it in есть:
        p = it["параметры104"]
        for цель, f in формула["поля"].items():
            xs = [float(p.get(k, 0) or 0) for k in f["из"]]
            v = sum(w * x for w, x in zip(f["веса"], xs)) + f["b"]
            p[цель] = round(float(np.clip(v, 0, 1)), 4)

    json.dump(d, open(кат, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(формула, open(формула_файл, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    ok = all(формула["поля"].get(c, {}).get("воспр_r", 0) >= 0.90 for c in цели)
    print("ГОТОВО: n_* записаны в параметры104.")
    print(f"формула: {формула_файл}")
    print("п104 можно не передавать в образ" if ok else "п104 оставить как fallback (r<0.90)")


if __name__ == "__main__":
    main()

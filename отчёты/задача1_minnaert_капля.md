# Задача 1 — Миннарт vs импульс-only

> эталон: `reference_drop.wav` · t_impact=2.2022 · seed=42

## Числа

| модель | band_corr | centroid_diff_hz | band_l2 |
|--------|----------:|-----------------:|--------:|
| impulse only | 0.866 | 1705.6 | 4.6507 |
| minnaert best | 0.8743 | 1006.2 | 3.8925 |

- bubble_freq ≈ **3260.0 Hz** · r_bubble=1.00 мм
- delay=1.0 ms · decay=18.0 ms · gain=1.2
- minnaert лучше band_corr: **True**
- minnaert лучше centroid: **True**

## Файлы
- эталон: `выход/причина_дождь/reference_drop.wav`
- synth: `выход/причина_дождь/synth_drop_minnaert.wav`
- impulse-only: `выход/причина_дождь/synth_drop_impulse_only.wav`

Слушай три файла. Число есть — гипотеза либо держится, либо нет.

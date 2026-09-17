# Синхрон-пилот дождь — инвентарь осей

> Дата: 2026-07-21 · путь-2 · только факты

| Ось | сек | заметка |
|-----|-----|---------|
| dozhd_real.wav | 2.4 | sr=22050 |
| video_live_02 | 9.9322 | fps=59.0 audio=True |
| clean_video | 9.9322 | fps=59.0 |
| atoms model3d span | 0.0 | n=19 max_birth=0.0 |
| atoms клетка span | 2.336 | n=7350 max_birth=2.336 |

## Флаги рассинхрона

- `mismatch_wav_vs_video_live_02`: **True**
- `mismatch_wav_vs_atom_span_model3d`: **True**
- `mismatch_video_vs_atom_span_model3d`: **True**

**Заметка:** Эталон dozhd_real.wav короче video_live_02 — нельзя считать их одной осью без явного правила.

JSON: `отчёты/синхрон_пилот_dozhd.json`

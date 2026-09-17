# -*- coding: utf-8 -*-
"""Единый источник порогов V3.3 — не дублировать в модулях."""
from __future__ import annotations

# --- фаза (ядро/фаза.py, узнавание) ---
HARMONICITY_ТОН = 0.6
HARMONICITY_ШУМ = 0.05
FLATNESS_ШУМ_КАДР = 0.3  # кадр → harmonicity=0 в HPS-пути

# --- атомизация: шум / тон ---
FLATNESS_ШУМ = 0.4       # кадр → 19 лог-полос
FLATNESS_ТОН_ЧИСТЫЙ = 0.15  # ниже → harmonicity=1.0 на пике
HARMONICITY_ПИК_СМЕШАН = 0.4

# --- пики ---
PEAK_AMP_MIN = 0.003
BAND_ENERGY_MIN = 0.0008
BAND_N = 19
BAND_F_LO = 60.0
BAND_F_HI = 9000.0

# --- треки ---
LOG2_FREQ_TOL = 1.0 / 12.0   # полутон
MAX_FRAME_GAP = 1            # кадров между точками трека
CLOSE_IDLE_FRAMES = 2        # без продолжения → закрыть

# --- STFT ---
SR = 22050
N_FFT = 2048
HOP = 512

# --- обогащение 104 ---
WINDOW_MS_MIN = 50
WINDOW_MS_MAX = 200

import os
import io
import json
import base64
import logging
import threading
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from functools import wraps
from flask import Flask, jsonify, request
from flask_cors import CORS

import firebase_admin
from firebase_admin import credentials, db, auth

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Base directory used for model and key paths
BASE = os.path.dirname(__file__)

# ── Testing Mode ──────────────────────────────────────────────────────────────
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
log.info(f"TEST_MODE={'ON' if TEST_MODE else 'OFF'}")

# ── Firebase init ─────────────────────────────────────────────────────────────
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", os.path.join(BASE, "firebase-key.json"))
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")

if TEST_MODE:
    log.warning("⚠️  Running in TEST MODE - Firebase disabled. Use /predict endpoint with dummy data.")
else:
    if not FIREBASE_DB_URL:
        raise RuntimeError("FIREBASE_DB_URL is required. Set it in your environment.")
    if not os.path.exists(FIREBASE_KEY_PATH):
        raise FileNotFoundError(
            f"Firebase service account key not found at '{FIREBASE_KEY_PATH}'. "
            "Set FIREBASE_KEY_PATH to a valid JSON key file."
        )

    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        "databaseURL": FIREBASE_DB_URL
    })

# ── Load sklearn models ───────────────────────────────────────────────────────
log.info("Loading sklearn models...")

# Disease XGBoost model kept loaded (used as a fallback tiebreaker only)
disease_model     = joblib.load(os.path.join(BASE, "models", "crop_disease_model.pkl"))
disease_le        = joblib.load(os.path.join(BASE, "models", "disease_label_encoder.pkl"))
crop_name_le      = joblib.load(os.path.join(BASE, "models", "crop_name_encoder.pkl"))

irrigation_model  = joblib.load(os.path.join(BASE, "models", "irrigation_model.pkl"))
irrigation_scaler = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
irrigation_le     = joblib.load(os.path.join(BASE, "models", "label_encoder.pkl"))

yield_model       = joblib.load(os.path.join(BASE, "models", "crop_yield_model.pkl"))

log.info("Sklearn models loaded.")

# ── Load CNN model ────────────────────────────────────────────────────────────
CNN_MODEL    = None
CNN_LABELS   = {}
CNN_IMG_SIZE = 224

CNN_MODEL_PATH  = os.path.join(BASE, "models", "plant_disease_cnn.h5")
CNN_LABELS_PATH = os.path.join(BASE, "models", "cnn_class_labels.json")

def load_cnn():
    global CNN_MODEL, CNN_LABELS
    try:
        import tensorflow as tf
        from tensorflow import keras
        CNN_MODEL  = keras.models.load_model(CNN_MODEL_PATH)
        with open(CNN_LABELS_PATH, "r") as f:
            CNN_LABELS = json.load(f)
        log.info("CNN model loaded — %d classes.", len(CNN_LABELS))
    except Exception as e:
        log.warning("CNN model not loaded: %s", e)

threading.Thread(target=load_cnn, daemon=True).start()

# ── CNN supported crops ───────────────────────────────────────────────────────
CNN_SUPPORTED_CROPS = {
    "apple", "blueberry", "cherry", "corn", "maize", "grape", "grapes",
    "orange", "peach", "bell pepper", "pepper", "potato", "raspberry",
    "soybean", "squash", "strawberry", "tomato",
    "rice", "wheat", "cotton", "banana", "mango"
}

# ── Crop name maps ────────────────────────────────────────────────────────────
DISEASE_CROPS = {
    "rice": "rice", "maize": "maize", "chickpea": "chickpea",
    "kidneybeans": "kidneybeans", "pigeonpeas": "pigeonpeas",
    "mothbeans": "mothbeans", "mungbean": "mungbean", "blackgram": "blackgram",
    "lentil": "lentil", "pomegranate": "pomegranate", "banana": "banana",
    "mango": "mango", "grapes": "grapes", "watermelon": "watermelon",
    "muskmelon": "muskmelon", "apple": "apple", "orange": "orange",
    "papaya": "papaya", "coconut": "coconut", "cotton": "cotton",
    "jute": "jute", "coffee": "coffee",
    "potato": "potato", "tomato": "tomato", "soybean": "soybean",
    "wheat": "wheat", "sugarcane": "sugarcane", "groundnut": "groundnut",
    "barley": "barley", "sorghum": "sorghum", "pea": "pea",
    "bean": "bean", "sweetpotato": "sweetpotato", "yam": "yam",
    "rapeseed": "rapeseed", "sugarbeet": "sugarbeet", "peach": "peach",
    "cherry": "cherry", "strawberry": "strawberry", "blueberry": "blueberry",
    "raspberry": "raspberry", "squash": "squash", "bellpepper": "bellpepper",
    "corn": "corn", "grape": "grape", "pepper": "pepper",
    "paddy": "paddy", "cassava": "cassava",
}

IRRIGATION_CROPS = {
    "wheat": "Wheat", "maize": "Maize", "rice": "Rice", "paddy": "Paddy",
    "potato": "Potato", "sugarcane": "Sugarcane", "coffee": "Coffee",
    "groundnuts": "Groundnuts", "pulse": "Pulse", "garden flowers": "Garden Flowers",
    "cotton": "Cotton", "banana": "Banana", "mango": "Mango", "tomato": "Tomato",
    "soybean": "Soybean", "onion": "Onion", "mustard": "Mustard",
    "sorghum": "Sorghum", "barley": "Barley", "chickpea": "Chickpea",
    "lentil": "Lentil", "sunflower": "Sunflower", "ginger": "Ginger",
    "turmeric": "Turmeric", "orange": "Orange",
}

YIELD_CROPS = {
    "banana": "Bananas", "barley": "Barley", "bean": "Beans",
    "cassava": "Cassava", "coffee": "Coffee", "cotton": "Cotton",
    "groundnut": "Groundnuts", "maize": "Maize", "orange": "Oranges",
    "palm oil": "Palm oil", "pea": "Peas", "plantain": "Plantains and others",
    "potato": "Potatoes", "rapeseed": "Rapeseed", "rice": "Rice",
    "sorghum": "Sorghum", "soybean": "Soybeans", "sugarbeet": "Sugarbeet",
    "sugarcane": "Sugarcane", "sweet potato": "Sweet potatoes",
    "tomato": "Tomatoes", "wheat": "Wheat", "yam": "Yams"
}

# ═════════════════════════════════════════════════════════════════════════════
# CROP-AWARE HEALTH RISK ENGINE
# Source: ICAR guidelines + FAO crop production thresholds
# Each crop has:
#   humidity_ideal   — normal operating range (min, max) %
#   humidity_danger  — above this = fungal/bacterial risk
#   rain_ideal       — normal annual rainfall range (mm)
#   rain_danger      — above this = waterlogging/fungal risk
#   temp_ideal       — optimal temperature range (°C)
#   temp_min         — frost/chilling injury below this
#   temp_max         — heat stress above this
#   ph_min / ph_max  — soil pH range (healthy)
#   n_max            — excess nitrogen threshold (kg/ha)
# ═════════════════════════════════════════════════════════════════════════════
CROP_HEALTH_PROFILES = {
    # ── Cereals & Staples ─────────────────────────────────────────────────────
    "rice": {
        # Rice blast (Magnaporthe oryzae) triggers at 90%+ humidity — ICAR threshold
        "humidity_ideal": (70, 90), "humidity_danger": 90,
        "rain_ideal": (150, 250),   "rain_danger": 350,
        "temp_ideal": (20, 35),     "temp_min": 15, "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 140,
    },
    "paddy": {
        "humidity_ideal": (70, 90), "humidity_danger": 90,
        "rain_ideal": (150, 250),   "rain_danger": 350,
        "temp_ideal": (20, 35),     "temp_min": 15, "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 140,
    },
    "wheat": {
        "humidity_ideal": (40, 70), "humidity_danger": 78,
        "rain_ideal": (30, 100),    "rain_danger": 150,
        "temp_ideal": (10, 25),     "temp_min": 2,  "temp_max": 32,
        "ph_min": 6.0, "ph_max": 7.5, "n_max": 120,
    },
    "maize": {
        "humidity_ideal": (50, 80), "humidity_danger": 85,
        "rain_ideal": (50, 200),    "rain_danger": 280,
        "temp_ideal": (18, 32),     "temp_min": 8,  "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 150,
    },
    "corn": {  # alias
        "humidity_ideal": (50, 80), "humidity_danger": 85,
        "rain_ideal": (50, 200),    "rain_danger": 280,
        "temp_ideal": (18, 32),     "temp_min": 8,  "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 150,
    },
    "barley": {
        "humidity_ideal": (40, 70), "humidity_danger": 78,
        "rain_ideal": (30, 100),    "rain_danger": 140,
        "temp_ideal": (10, 25),     "temp_min": 0,  "temp_max": 30,
        "ph_min": 6.0, "ph_max": 8.0, "n_max": 100,
    },
    "sorghum": {
        "humidity_ideal": (35, 70), "humidity_danger": 80,
        "rain_ideal": (40, 180),    "rain_danger": 250,
        "temp_ideal": (20, 38),     "temp_min": 10, "temp_max": 42,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 120,
    },
    # ── Pulses ────────────────────────────────────────────────────────────────
    "chickpea": {
        "humidity_ideal": (30, 60), "humidity_danger": 72,
        "rain_ideal": (20, 100),    "rain_danger": 140,
        "temp_ideal": (15, 30),     "temp_min": 5,  "temp_max": 35,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 60,
    },
    "lentil": {
        "humidity_ideal": (30, 65), "humidity_danger": 75,
        "rain_ideal": (20, 100),    "rain_danger": 140,
        "temp_ideal": (12, 28),     "temp_min": 5,  "temp_max": 32,
        "ph_min": 6.0, "ph_max": 8.0, "n_max": 60,
    },
    "pea": {
        "humidity_ideal": (40, 70), "humidity_danger": 80,
        "rain_ideal": (30, 120),    "rain_danger": 180,
        "temp_ideal": (10, 25),     "temp_min": 2,  "temp_max": 30,
        "ph_min": 6.0, "ph_max": 7.5, "n_max": 80,
    },
    "bean": {
        "humidity_ideal": (40, 75), "humidity_danger": 85,
        "rain_ideal": (40, 160),    "rain_danger": 220,
        "temp_ideal": (15, 28),     "temp_min": 8,  "temp_max": 32,
        "ph_min": 6.0, "ph_max": 7.5, "n_max": 80,
    },
    "blackgram": {
        "humidity_ideal": (50, 80), "humidity_danger": 88,
        "rain_ideal": (60, 180),    "rain_danger": 250,
        "temp_ideal": (25, 35),     "temp_min": 15, "temp_max": 40,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 80,
    },
    "mungbean": {
        "humidity_ideal": (50, 80), "humidity_danger": 88,
        "rain_ideal": (60, 180),    "rain_danger": 250,
        "temp_ideal": (25, 35),     "temp_min": 15, "temp_max": 40,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 80,
    },
    "mothbeans": {
        "humidity_ideal": (30, 65), "humidity_danger": 78,
        "rain_ideal": (30, 120),    "rain_danger": 160,
        "temp_ideal": (25, 38),     "temp_min": 15, "temp_max": 42,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 60,
    },
    "pigeonpeas": {
        "humidity_ideal": (50, 80), "humidity_danger": 88,
        "rain_ideal": (60, 200),    "rain_danger": 280,
        "temp_ideal": (20, 35),     "temp_min": 12, "temp_max": 40,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 80,
    },
    "kidneybeans": {
        "humidity_ideal": (40, 75), "humidity_danger": 85,
        "rain_ideal": (40, 160),    "rain_danger": 220,
        "temp_ideal": (15, 28),     "temp_min": 8,  "temp_max": 32,
        "ph_min": 6.0, "ph_max": 7.5, "n_max": 80,
    },
    # ── Cash Crops ────────────────────────────────────────────────────────────
    "cotton": {
        "humidity_ideal": (50, 75), "humidity_danger": 85,
        "rain_ideal": (60, 180),    "rain_danger": 250,
        "temp_ideal": (20, 35),     "temp_min": 12, "temp_max": 40,
        "ph_min": 5.8, "ph_max": 7.5, "n_max": 120,
    },
    "sugarcane": {
        "humidity_ideal": (65, 85), "humidity_danger": 92,
        "rain_ideal": (100, 250),   "rain_danger": 320,
        "temp_ideal": (20, 38),     "temp_min": 12, "temp_max": 42,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 200,
    },
    "jute": {
        # Jute LOVES humidity — normal range is 75-95%
        "humidity_ideal": (75, 95), "humidity_danger": 99,
        "rain_ideal": (150, 350),   "rain_danger": 450,
        "temp_ideal": (25, 38),     "temp_min": 18, "temp_max": 42,
        "ph_min": 5.0, "ph_max": 7.5, "n_max": 100,
    },
    "coffee": {
        # Coffee is subtropical — 70-90% humidity is NORMAL, not dangerous
        "humidity_ideal": (65, 90), "humidity_danger": 97,
        "rain_ideal": (120, 250),   "rain_danger": 320,
        "temp_ideal": (18, 28),     "temp_min": 10, "temp_max": 32,
        "ph_min": 5.5, "ph_max": 6.5, "n_max": 100,
    },
    "rapeseed": {
        "humidity_ideal": (40, 70), "humidity_danger": 78,
        "rain_ideal": (30, 120),    "rain_danger": 160,
        "temp_ideal": (8, 22),      "temp_min": 0,  "temp_max": 28,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 120,
    },
    "sugarbeet": {
        "humidity_ideal": (40, 70), "humidity_danger": 80,
        "rain_ideal": (40, 150),    "rain_danger": 200,
        "temp_ideal": (10, 25),     "temp_min": 2,  "temp_max": 32,
        "ph_min": 6.5, "ph_max": 8.0, "n_max": 130,
    },
    "groundnut": {
        "humidity_ideal": (50, 75), "humidity_danger": 85,
        "rain_ideal": (50, 180),    "rain_danger": 250,
        "temp_ideal": (22, 35),     "temp_min": 15, "temp_max": 40,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 80,
    },
    # ── Fruits ────────────────────────────────────────────────────────────────
    "banana": {
        # Banana is tropical — 75-90% humidity is NORMAL
        # Rain ideal_max tightened: >220mm/month = waterlogging/Fusarium risk
        "humidity_ideal": (70, 90), "humidity_danger": 97,
        "rain_ideal": (100, 220),   "rain_danger": 320,
        "temp_ideal": (22, 35),     "temp_min": 14, "temp_max": 40,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 200,
    },
    "mango": {
        "humidity_ideal": (50, 80), "humidity_danger": 90,
        "rain_ideal": (50, 200),    "rain_danger": 280,
        "temp_ideal": (24, 38),     "temp_min": 10, "temp_max": 44,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 100,
    },
    "orange": {
        "humidity_ideal": (50, 80), "humidity_danger": 88,
        "rain_ideal": (60, 180),    "rain_danger": 250,
        "temp_ideal": (15, 32),     "temp_min": 5,  "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 120,
    },
    "apple": {
        "humidity_ideal": (50, 75), "humidity_danger": 82,
        "rain_ideal": (60, 160),    "rain_danger": 220,
        "temp_ideal": (8, 22),      "temp_min": -2, "temp_max": 28,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 100,
    },
    "grapes": {
        "humidity_ideal": (40, 70), "humidity_danger": 78,
        "rain_ideal": (30, 120),    "rain_danger": 180,
        "temp_ideal": (15, 32),     "temp_min": 0,  "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 80,
    },
    "grape": {  # alias
        "humidity_ideal": (40, 70), "humidity_danger": 78,
        "rain_ideal": (30, 120),    "rain_danger": 180,
        "temp_ideal": (15, 32),     "temp_min": 0,  "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 80,
    },
    "pomegranate": {
        "humidity_ideal": (35, 65), "humidity_danger": 78,
        "rain_ideal": (30, 120),    "rain_danger": 170,
        "temp_ideal": (20, 38),     "temp_min": 5,  "temp_max": 44,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 80,
    },
    "watermelon": {
        "humidity_ideal": (50, 75), "humidity_danger": 85,
        "rain_ideal": (40, 150),    "rain_danger": 200,
        "temp_ideal": (22, 35),     "temp_min": 15, "temp_max": 40,
        "ph_min": 6.0, "ph_max": 7.5, "n_max": 80,
    },
    "muskmelon": {
        "humidity_ideal": (45, 75), "humidity_danger": 85,
        "rain_ideal": (40, 140),    "rain_danger": 190,
        "temp_ideal": (22, 35),     "temp_min": 15, "temp_max": 40,
        "ph_min": 6.0, "ph_max": 7.5, "n_max": 80,
    },
    "coconut": {
        # Coconut is tropical coastal — 80-95% humidity is COMPLETELY NORMAL
        "humidity_ideal": (75, 95), "humidity_danger": 99,
        "rain_ideal": (100, 300),   "rain_danger": 400,
        "temp_ideal": (25, 38),     "temp_min": 15, "temp_max": 42,
        "ph_min": 5.0, "ph_max": 8.0, "n_max": 100,
    },
    "papaya": {
        # Papaya tropical — 65-90% humidity is NORMAL
        "humidity_ideal": (65, 90), "humidity_danger": 96,
        "rain_ideal": (100, 250),   "rain_danger": 330,
        "temp_ideal": (22, 35),     "temp_min": 12, "temp_max": 40,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 120,
    },
    "peach": {
        "humidity_ideal": (45, 72), "humidity_danger": 80,
        "rain_ideal": (40, 140),    "rain_danger": 200,
        "temp_ideal": (8, 25),      "temp_min": -5, "temp_max": 32,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 80,
    },
    "cherry": {
        "humidity_ideal": (45, 72), "humidity_danger": 80,
        "rain_ideal": (40, 140),    "rain_danger": 200,
        "temp_ideal": (8, 22),      "temp_min": -5, "temp_max": 28,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 80,
    },
    "strawberry": {
        "humidity_ideal": (50, 75), "humidity_danger": 85,
        "rain_ideal": (40, 140),    "rain_danger": 200,
        "temp_ideal": (12, 26),     "temp_min": 0,  "temp_max": 32,
        "ph_min": 5.5, "ph_max": 6.5, "n_max": 80,
    },
    "blueberry": {
        "humidity_ideal": (50, 75), "humidity_danger": 85,
        "rain_ideal": (50, 160),    "rain_danger": 220,
        "temp_ideal": (10, 25),     "temp_min": -5, "temp_max": 30,
        "ph_min": 4.5, "ph_max": 5.5, "n_max": 60,
    },
    "raspberry": {
        "humidity_ideal": (50, 75), "humidity_danger": 85,
        "rain_ideal": (50, 160),    "rain_danger": 220,
        "temp_ideal": (10, 25),     "temp_min": -5, "temp_max": 30,
        "ph_min": 5.5, "ph_max": 6.5, "n_max": 80,
    },
    # ── Vegetables ────────────────────────────────────────────────────────────
    "potato": {
        # Late blight (Phytophthora infestans) triggers at 85%+ humidity — NOT 88
        "humidity_ideal": (60, 80), "humidity_danger": 85,
        "rain_ideal": (50, 150),    "rain_danger": 200,
        "temp_ideal": (10, 24),     "temp_min": 2,  "temp_max": 30,
        "ph_min": 5.0, "ph_max": 6.5, "n_max": 120,
    },
    "tomato": {
        "humidity_ideal": (55, 80), "humidity_danger": 88,
        "rain_ideal": (40, 150),    "rain_danger": 220,
        "temp_ideal": (18, 32),     "temp_min": 8,  "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 120,
    },
    "soybean": {
        "humidity_ideal": (50, 80), "humidity_danger": 88,
        "rain_ideal": (60, 200),    "rain_danger": 280,
        "temp_ideal": (20, 32),     "temp_min": 10, "temp_max": 38,
        "ph_min": 5.8, "ph_max": 7.0, "n_max": 80,
    },
    "pepper": {
        "humidity_ideal": (55, 80), "humidity_danger": 88,
        "rain_ideal": (50, 160),    "rain_danger": 220,
        "temp_ideal": (18, 32),     "temp_min": 10, "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 100,
    },
    "bellpepper": {
        "humidity_ideal": (55, 80), "humidity_danger": 88,
        "rain_ideal": (50, 160),    "rain_danger": 220,
        "temp_ideal": (18, 32),     "temp_min": 10, "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 100,
    },
    "squash": {
        "humidity_ideal": (50, 78), "humidity_danger": 88,
        "rain_ideal": (50, 160),    "rain_danger": 220,
        "temp_ideal": (18, 32),     "temp_min": 10, "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 100,
    },
    # ── Root Crops ────────────────────────────────────────────────────────────
    "cassava": {
        "humidity_ideal": (60, 85), "humidity_danger": 93,
        "rain_ideal": (80, 250),    "rain_danger": 350,
        "temp_ideal": (20, 35),     "temp_min": 12, "temp_max": 40,
        "ph_min": 5.0, "ph_max": 7.5, "n_max": 100,
    },
    "yam": {
        "humidity_ideal": (65, 88), "humidity_danger": 95,
        "rain_ideal": (100, 280),   "rain_danger": 380,
        "temp_ideal": (22, 35),     "temp_min": 15, "temp_max": 40,
        "ph_min": 5.5, "ph_max": 7.5, "n_max": 100,
    },
    "sweetpotato": {
        "humidity_ideal": (55, 80), "humidity_danger": 88,
        "rain_ideal": (50, 180),    "rain_danger": 250,
        "temp_ideal": (20, 32),     "temp_min": 12, "temp_max": 38,
        "ph_min": 5.5, "ph_max": 7.0, "n_max": 80,
    },
}


def crop_health_risk_engine(crop, N, P, K, temp, humidity, rainfall, ph):
    """
    Crop-aware rule engine replacing XGBoost disease model.
    Returns same interface as old predict_disease() for zero breaking changes.

    Scoring:
        0–29  → Healthy
        30–49 → Healthy (borderline, monitor)
        50+   → At_Risk

    Each risk factor contributes a weighted score based on how far
    the reading deviates from the crop's known safe range.
    """
    profile = CROP_HEALTH_PROFILES.get(crop)

    # Fallback: if crop not profiled, use conservative global thresholds
    if profile is None:
        profile = {
            "humidity_ideal": (50, 80), "humidity_danger": 88,
            "rain_ideal": (50, 200),    "rain_danger": 300,
            "temp_ideal": (15, 35),     "temp_min": 5, "temp_max": 42,
            "ph_min": 5.5, "ph_max": 7.5, "n_max": 130,
        }

    risk_score = 0
    reasons    = []

    # ── 1. Humidity check (weight: 30) ────────────────────────────────────────
    h_ideal_min, h_ideal_max = profile["humidity_ideal"]
    h_danger = profile["humidity_danger"]

    if humidity > h_danger:
        risk_score += 30
        reasons.append(f"Humidity {humidity}% is dangerously high (danger threshold: {h_danger}%)")
    elif humidity > h_ideal_max:
        # Proportional score between ideal_max and danger
        ratio = (humidity - h_ideal_max) / max(h_danger - h_ideal_max, 1)
        risk_score += int(ratio * 20)
        reasons.append(f"Humidity {humidity}% slightly above ideal range ({h_ideal_min}–{h_ideal_max}%)")
    elif humidity < h_ideal_min:
        risk_score += 5
        reasons.append(f"Humidity {humidity}% below ideal minimum ({h_ideal_min}%)")

    # ── 2. Rainfall check (weight: 25) ────────────────────────────────────────
    r_ideal_min, r_ideal_max = profile["rain_ideal"]
    r_danger = profile["rain_danger"]

    if rainfall > r_danger:
        risk_score += 25
        reasons.append(f"Rainfall {rainfall}mm exceeds danger threshold ({r_danger}mm) — waterlogging/fungal risk")
    elif rainfall > r_ideal_max:
        ratio = (rainfall - r_ideal_max) / max(r_danger - r_ideal_max, 1)
        risk_score += int(ratio * 15)
        reasons.append(f"Rainfall {rainfall}mm above ideal range ({r_ideal_min}–{r_ideal_max}mm)")
    elif rainfall < r_ideal_min:
        risk_score += 8
        reasons.append(f"Rainfall {rainfall}mm below ideal minimum ({r_ideal_min}mm) — drought stress")

    # ── 3. Temperature check (weight: 20) ─────────────────────────────────────
    t_min = profile["temp_min"]
    t_max = profile["temp_max"]
    t_ideal_min, t_ideal_max = profile["temp_ideal"]

    if temp > t_max:
        risk_score += 20
        reasons.append(f"Temperature {temp}°C above safe maximum ({t_max}°C) — heat stress")
    elif temp < t_min:
        risk_score += 20
        reasons.append(f"Temperature {temp}°C below safe minimum ({t_min}°C) — frost/chilling injury")
    elif temp > t_ideal_max:
        ratio = (temp - t_ideal_max) / max(t_max - t_ideal_max, 1)
        risk_score += int(ratio * 12)
        reasons.append(f"Temperature {temp}°C slightly above optimal range")
    elif temp < t_ideal_min:
        ratio = (t_ideal_min - temp) / max(t_ideal_min - t_min, 1)
        risk_score += int(ratio * 12)
        reasons.append(f"Temperature {temp}°C slightly below optimal range")

    # ── 4. pH check (weight: 15) ──────────────────────────────────────────────
    if ph < profile["ph_min"]:
        severity = profile["ph_min"] - ph
        risk_score += min(15, int(severity * 8))
        reasons.append(f"Soil pH {ph} below optimal minimum ({profile['ph_min']}) — nutrient lockout risk")
    elif ph > profile["ph_max"]:
        severity = ph - profile["ph_max"]
        risk_score += min(15, int(severity * 8))
        reasons.append(f"Soil pH {ph} above optimal maximum ({profile['ph_max']}) — alkalinity stress")

    # ── 5. Nitrogen excess check (weight: 10) ─────────────────────────────────
    n_max = profile.get("n_max", 130)
    n_min = profile.get("n_min", 20)   # plants with very low N = weakened immunity
    if N > n_max:
        risk_score += min(10, int(((N - n_max) / n_max) * 10))
        reasons.append(f"N={N} kg/ha exceeds safe threshold ({n_max}) — lush growth increases disease susceptibility")
    elif N < n_min:
        risk_score += 10
        reasons.append(f"N={N} kg/ha critically low ({n_min}) — nutrient deficiency weakens plant immunity")

    # ── 6. Compound multi-stress bonus (weight: +15) ──────────────────────────
    # Agronomic reality: co-occurring stresses are not additive, they're synergistic.
    # Cold + wet + acidic soil simultaneously = exponentially higher disease risk.
    # This mirrors ICAR's disease forecasting models which flag multi-factor alerts.
    if len(reasons) >= 3:
        risk_score += 20
        reasons.append("Multiple simultaneous stress factors detected — compound risk elevated")

    # ── Decision ──────────────────────────────────────────────────────────────
    risk_score = min(risk_score, 100)  # cap at 100

    if risk_score >= 50:
        label = "At_Risk"
    else:
        label = "Healthy"

    # Convert to probability-like values for dashboard compatibility
    at_risk_prob  = round(risk_score / 100, 4)
    healthy_prob  = round(1 - at_risk_prob, 4)

    return {
        "label"      : label,
        "atRiskProb" : at_risk_prob,
        "healthyProb": healthy_prob,
        "riskScore"  : risk_score,          # NEW — 0 to 100
        "riskReasons": reasons,             # NEW — explainable reasons list
        "diseaseName": None,
        "timestamp"  : datetime.utcnow().isoformat()
    }


# ── YIELD CROP NAME VALIDATOR ─────────────────────────────────────────────────
# The repeating 8,450 bug is caused by the OneHotEncoder silently mapping
# unknown/mismatched Item names to a default column during transform.
# This validator ensures the exact string the model was trained on is used.

YIELD_ITEM_EXACT = {v: v for v in YIELD_CROPS.values()}  # ground truth strings

def _safe_yield_item(crop_key):
    """
    Returns the exact Item string the yield model's OneHotEncoder was trained on.
    Falls back to None if unrecognised, which we catch before calling the model.
    """
    return YIELD_CROPS.get(crop_key.lower().strip())


app = Flask(__name__)


def _cors_allowed_origins():
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS(
    app,
    resources={r"/*": {"origins": _cors_allowed_origins()}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

log.info("CORS enabled for origins: %s", ", ".join(_cors_allowed_origins()))

# ── Sensor Data Validation ────────────────────────────────────────────────────
# Define valid ranges for each sensor
SENSOR_RANGES = {
    "temperature": (-50, 60),           # °C
    "humidity": (0, 100),               # %RH
    "soilMoisture": (100, 1000),        # 0-1000 scale (dry to wet)
    "rainfall": (0, 5000),              # mm
    "pH": (0, 14),                      # 0-14 scale
    "N": (0, 500),                      # mg/kg nitrogen
    "P": (0, 500),                      # mg/kg phosphorus
    "K": (0, 500)                       # mg/kg potassium
}

# Crop-specific sensor thresholds (optional, more accurate)
CROP_SPECIFIC_RANGES = {
    "rice": {"N": (20, 200), "pH": (5.5, 7.0), "temperature": (15, 38)},
    "wheat": {"N": (60, 140), "pH": (6.0, 7.5), "temperature": (2, 32)},
    "maize": {"N": (40, 120), "pH": (6.0, 7.0), "temperature": (10, 35)},
    "cotton": {"N": (40, 120), "pH": (6.0, 7.0), "temperature": (15, 40)},
    "tomato": {"N": (80, 200), "pH": (6.0, 6.8), "temperature": (20, 35)},
}

def validate_sensor_data(sensor, crop=None):
    """Validate sensor data is within acceptable ranges.
    
    Returns: (valid, errors_list)
    """
    errors = []
    
    if not isinstance(sensor, dict):
        errors.append("sensor must be a dict")
        return False, errors
    
    # Check each sensor
    for key, (min_val, max_val) in SENSOR_RANGES.items():
        if key not in sensor:
            errors.append(f"Missing {key}")
            continue
        
        val = sensor[key]
        if not isinstance(val, (int, float)):
            errors.append(f"{key} must be numeric")
            continue
        
        # Use crop-specific range if available
        if crop and crop in CROP_SPECIFIC_RANGES and key in CROP_SPECIFIC_RANGES[crop]:
            min_val, max_val = CROP_SPECIFIC_RANGES[crop][key]
        
        if val < min_val or val > max_val:
            errors.append(f"{key}={val} out of range [{min_val}, {max_val}]")
    
    return len(errors) == 0, errors

def validate_config_data(config):
    """Validate config data structure and values."""
    errors = []
    
    if not isinstance(config, dict):
        errors.append("config must be a dict")
        return False, errors
    
    # Check required fields
    required = ["cropType"]
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate cropType
    if "cropType" in config:
        crop = config["cropType"].lower()
        if not crop or len(crop) > 50:
            errors.append(f"cropType invalid: {config['cropType']}")
    
    # Validate numeric fields
    if "days" in config:
        if not isinstance(config["days"], int) or not (1 <= config["days"] <= 365):
            errors.append(f"days must be 1-365, got {config.get('days')}")
    
    if "year" in config:
        if not isinstance(config["year"], int) or not (1990 <= config["year"] <= 2100):
            errors.append(f"year must be 1990-2100, got {config.get('year')}")
    
    if "pesticides" in config:
        if not isinstance(config["pesticides"], int) or config["pesticides"] < 0:
            errors.append(f"pesticides must be >= 0, got {config.get('pesticides')}")
    
    return len(errors) == 0, errors


# ── Auth decorator ────────────────────────────────────────────────────────────
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # In TEST_MODE, use dummy user ID
        if TEST_MODE:
            request.uid = "test_user_001"
            request.user_email = "test@kisaan.ai"
            return f(*args, **kwargs)
        
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        id_token = auth_header[len("Bearer "):].strip()
        try:
            decoded = auth.verify_id_token(id_token)
            request.uid = decoded["uid"]
            request.user_email = decoded.get("email", "unknown")
        except Exception as e:
            log.warning("Token verification failed: %s", e)
            return jsonify({"error": "Invalid or expired token"}), 401
        return f(*args, **kwargs)
    return decorated


# ── Firebase helpers ──────────────────────────────────────────────────────────
def user_sensors_ref(uid):
    if TEST_MODE:
        return None
    return db.reference(f"/users/{uid}/sensors/latest")

def user_predictions_ref(uid):
    if TEST_MODE:
        return None
    return db.reference(f"/users/{uid}/predictions")

def user_config_ref(uid):
    if TEST_MODE:
        return None
    return db.reference(f"/users/{uid}/config")

def user_cnn_ref(uid):
    if TEST_MODE:
        return None
    return db.reference(f"/users/{uid}/cnn_prediction")


def parse_json_body():
    if not request.is_json:
        return None, (jsonify({"error": "Request content-type must be application/json"}), 415)
    body = request.get_json(silent=True)
    if body is None:
        return None, (jsonify({"error": "Malformed JSON payload"}), 400)
    return body, None


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def predict_disease(sensor, config):
    """
    Crop-aware health risk engine.
    Replaces XGBoost model entirely. No .pkl file needed for disease prediction.
    Returns same response shape as before for zero dashboard changes.
    """
    crop = config.get("cropType", "").lower().strip()

    if crop not in DISEASE_CROPS:
        return {
            "label"      : "Not Available",
            "atRiskProb" : None,
            "healthyProb": None,
            "riskScore"  : None,
            "riskReasons": [],
            "reason"     : f"'{crop}' not supported.",
            "supported"  : sorted(DISEASE_CROPS.keys()),
            "timestamp"  : datetime.utcnow().isoformat()
        }

    N    = float(sensor.get("N", 0))
    P    = float(sensor.get("P", 0))
    K    = float(sensor.get("K", 0))
    temp = float(sensor.get("temperature", 25))
    rh   = float(sensor.get("humidity", 60))
    rain = float(sensor.get("rainfall", 100))
    ph   = float(sensor.get("ph", 6.5))

    result = crop_health_risk_engine(crop, N, P, K, temp, rh, rain, ph)
    log.info("Health risk [%s] → %s (score=%d)", crop, result["label"], result["riskScore"])
    return result


def predict_irrigation(sensor, config):
    crop = config.get("cropType", "").lower().strip()
    if crop not in IRRIGATION_CROPS:
        return {
            "irrigate"  : None,
            "label"     : "Not Available",
            "confidence": None,
            "reason"    : f"'{crop}' not supported.",
            "supported" : sorted(IRRIGATION_CROPS.keys()),
            "timestamp" : datetime.utcnow().isoformat()
        }

    crop_enc = irrigation_le.transform([IRRIGATION_CROPS[crop]])[0]
    X_raw = pd.DataFrame([{
        "CropType_enc": int(crop_enc),
        "CropDays"    : int(config.get("cropDays", 30)),
        "SoilMoisture": float(sensor["soilMoisture"]),
        "temperature" : float(sensor["temperature"]),
        "Humidity"    : float(sensor["humidity"])
    }])
    X_scaled = irrigation_scaler.transform(X_raw)
    pred = irrigation_model.predict(X_scaled)[0]
    prob = irrigation_model.predict_proba(X_scaled)[0]
    soil = float(sensor["soilMoisture"])

    # Hard rules: extreme soil moisture values always override model
    # Scale: 120=wet, 800=dry
    if soil >= 550:
        return {"irrigate": 1, "label": "Irrigate",
                "confidence": round(float(max(prob)), 4),
                "timestamp": datetime.utcnow().isoformat()}
    elif soil <= 420:   # tightened — soil <=420 is wet enough, no irrigation needed
        return {"irrigate": 0, "label": "No Irrigation",
                "confidence": round(float(max(prob)), 4),
                "timestamp": datetime.utcnow().isoformat()}

    # Middle zone (350–550): trust the model
    return {
        "irrigate"  : int(pred),
        "label"     : "Irrigate" if pred == 1 else "No Irrigation",
        "confidence": round(float(max(prob)), 4),
        "timestamp" : datetime.utcnow().isoformat()
    }


def predict_yield(sensor, config):
    crop = config.get("cropType", "").lower().strip()

    # ── Guard: verify crop is supported ──────────────────────────────────────
    item_name = _safe_yield_item(crop)
    if item_name is None:
        return {
            "hgPerHa"  : None,
            "kgPerHa"  : None,
            "reason"   : f"'{crop}' not supported.",
            "supported": sorted(YIELD_CROPS.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }

    # ── Build feature row with EXACT column values model was trained on ───────
    country   = config.get("country", "India")
    year      = int(config.get("year", datetime.utcnow().year))
    rainfall  = max(1.0, float(sensor.get("rainfall", 100)))
    pesticides = float(config.get("pesticides", 100))
    avg_temp  = float(sensor.get("temperature", 25))

    X = pd.DataFrame([{
        "Area"                          : country,
        "Item"                          : item_name,   # exact trained string
        "Year"                          : year,
        "average_rain_fall_mm_per_year" : rainfall,
        "pesticides_tonnes"             : pesticides,
        "avg_temp"                      : avg_temp
    }])

    # ── Predict and sanity-check result ──────────────────────────────────────
    try:
        hg_per_ha = float(yield_model.predict(X)[0])
    except Exception as e:
        log.error("Yield model predict error for crop '%s' (item='%s'): %s", crop, item_name, e)
        return {"hgPerHa": None, "kgPerHa": None, "error": str(e),
                "timestamp": datetime.utcnow().isoformat()}

    # Sanity check: most crops yield between 500 and 1,000,000 hg/ha
    # If the model returns something clearly wrong, log a warning
    if not (500 <= hg_per_ha <= 1_000_000):
        log.warning("Yield model returned suspicious value for '%s': %.0f hg/ha "
                    "(item='%s', country='%s', year=%d, rain=%.1f, temp=%.1f)",
                    crop, hg_per_ha, item_name, country, year, rainfall, avg_temp)

    return {
        "hgPerHa"  : round(hg_per_ha, 2),
        "kgPerHa"  : round(hg_per_ha / 10, 2),
        "timestamp": datetime.utcnow().isoformat()
    }


def run_all_predictions(sensor, config):
    results = {}
    crop = config.get("cropType", "unknown").lower()
    log.info("Running predictions for crop: '%s'", crop)
    for name, fn in [
        ("disease",    predict_disease),
        ("irrigation", predict_irrigation),
        ("yield",      predict_yield)
    ]:
        try:
            results[name] = fn(sensor, config)
            log.info("%s → %s", name.capitalize(),
                     results[name].get("label", results[name].get("kgPerHa", "?")))
        except Exception as e:
            log.error("%s prediction error: %s", name, e)
            results[name] = {"label": "Error", "error": str(e)}
    return results


# ── CNN prediction function ───────────────────────────────────────────────────
def predict_leaf_image(image_b64):
    if CNN_MODEL is None:
        return {"error": "CNN model not loaded. Check plant_disease_cnn.h5 in backend/models/"}
    try:
        from PIL import Image
        import tensorflow as tf

        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((CNN_IMG_SIZE, CNN_IMG_SIZE))

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = CNN_MODEL.predict(arr, verbose=0)[0]
        top3_indices = np.argsort(preds)[::-1][:3]

        top3 = []
        for idx in top3_indices:
            info = CNN_LABELS.get(str(idx), {})
            top3.append({
                "crop"      : info.get("crop", "Unknown"),
                "disease"   : info.get("disease", "Unknown"),
                "is_healthy": info.get("is_healthy", False),
                "confidence": round(float(preds[idx]) * 100, 2),
                "treatment" : info.get("treatment", "Consult local agronomist.")
            })

        best = top3[0]
        return {
            "crop"      : best["crop"],
            "disease"   : best["disease"],
            "is_healthy": best["is_healthy"],
            "confidence": best["confidence"],
            "treatment" : best["treatment"],
            "top3"      : top3,
            "timestamp" : datetime.utcnow().isoformat()
        }
    except Exception as e:
        log.error("CNN prediction error: %s", e)
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


# ── Firebase listener ─────────────────────────────────────────────────────────
ACTIVE_LISTENERS = {}
LISTENER_LOCK = threading.Lock()
MAX_USER_LISTENERS = int(os.getenv("MAX_USER_LISTENERS", "5000"))


def on_user_sensor_update(uid):
    def handler(event):
        if event.data is None:
            return
        log.info("Sensor update for uid=%s", uid)
        try:
            sensor = event.data
            config = user_config_ref(uid).get() or {}
            if not config:
                log.warning("No config for uid=%s — skipping prediction.", uid)
                return
            predictions = run_all_predictions(sensor, config)
            if not TEST_MODE:
                user_predictions_ref(uid).set(predictions)
            log.info("Predictions written for uid=%s", uid)
        except Exception as e:
            log.error("Listener error for uid=%s: %s", uid, e)
    return handler


def register_listener_for_user(uid):
    with LISTENER_LOCK:
        if uid in ACTIVE_LISTENERS:
            log.info("Listener already active for uid=%s", uid)
            return False
        if len(ACTIVE_LISTENERS) >= MAX_USER_LISTENERS:
            raise RuntimeError("Listener limit reached; cannot register more users.")

        stream = user_sensors_ref(uid).listen(on_user_sensor_update(uid))
        ACTIVE_LISTENERS[uid] = stream
        log.info("Firebase listener registered for uid=%s", uid)
        return True


def start_firebase_listener():
    def on_new_user(event):
        if event.data is None:
            return
        uid = event.path.strip("/").split("/")[0]
        if uid:
            try:
                register_listener_for_user(uid)
            except Exception as e:
                log.warning("Failed to register listener for uid=%s: %s", uid, e)
    db.reference("/users").listen(on_new_user)
    log.info("Master listener started — watching /users/ for new farmers.")


# ── REST Endpoints ────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@require_auth
def predict_endpoint():
    body, err = parse_json_body()
    if err:
        return err
    sensor = body.get("sensor", {})
    config = body.get("config", {})
    
    if not sensor or not config:
        return jsonify({"error": "Missing sensor or config"}), 400
    
    # Validate sensor data
    sensor_valid, sensor_errors = validate_sensor_data(sensor, config.get("cropType"))
    if not sensor_valid:
        log.warning("Sensor validation failed: %s", sensor_errors)
        return jsonify({
            "error": "Invalid sensor data",
            "details": sensor_errors
        }), 400
    
    # Validate config data
    config_valid, config_errors = validate_config_data(config)
    if not config_valid:
        log.warning("Config validation failed: %s", config_errors)
        return jsonify({
            "error": "Invalid config data",
            "details": config_errors
        }), 400
    
    results = run_all_predictions(sensor, config)
    
    # Only write to Firebase if not in TEST_MODE
    if not TEST_MODE:
        user_predictions_ref(request.uid).set(results)
    
    log.info("Manual predict called by %s for crop=%s", request.user_email, config.get("cropType"))
    return jsonify(results), 200


@app.route("/predict/now", methods=["GET"])
@require_auth
def predict_now():
    uid = request.uid
    
    # In TEST_MODE, use dummy data
    if TEST_MODE:
        sensor = {
            "temperature": 28.5, "humidity": 72, "soilMoisture": 450,
            "rainfall": 85, "pH": 6.8, "N": 45, "P": 30, "K": 25
        }
        config = {
            "cropType": "rice", "country": "India", "days": 60,
            "pesticides": 2, "year": 2024
        }
    else:
        sensor = user_sensors_ref(uid).get()
        config = user_config_ref(uid).get()
        if not sensor:
            return jsonify({"error": f"No sensor data at /users/{uid}/sensors/latest"}), 404
        if not config:
            return jsonify({"error": f"No config at /users/{uid}/config"}), 404
    
    results = run_all_predictions(sensor, config)
    if not TEST_MODE:
        user_predictions_ref(uid).set(results)
    log.info("predict/now called by %s", request.user_email)
    return jsonify(results), 200


@app.route("/predict/image", methods=["POST"])
@require_auth
def predict_image():
    body, err = parse_json_body()
    if err:
        return err
    image = body.get("image", "")
    if not image:
        return jsonify({"error": "Missing 'image' field in request body"}), 400
    if len(image) > 10 * 1024 * 1024:
        return jsonify({"error": "Image too large. Max 10MB."}), 400
    result = predict_leaf_image(image)
    if "error" not in result:
        if not TEST_MODE:
            user_cnn_ref(request.uid).set(result)
        log.info("CNN prediction for %s → %s (%s%%)",
                 request.user_email, result.get("disease"), result.get("confidence"))
    else:
        log.error("CNN prediction failed for %s: %s", request.user_email, result["error"])
    return jsonify(result), 200 if "error" not in result else 500


@app.route("/crops", methods=["GET"])
def supported_crops():
    return jsonify({
        "disease_model"   : sorted(DISEASE_CROPS.keys()),
        "irrigation_model": sorted(IRRIGATION_CROPS.keys()),
        "yield_model"     : sorted(YIELD_CROPS.keys()),
        "cnn_scanner"     : sorted(CNN_SUPPORTED_CROPS),
        "works_in_all_3"  : sorted(set(DISEASE_CROPS) & set(IRRIGATION_CROPS) & set(YIELD_CROPS))
    }), 200


@app.route("/status", methods=["GET"])
@require_auth
def status():
    return jsonify({
        "status" : "running",
        "models" : {
            "disease"   : "rule_engine_v2",   # updated label
            "irrigation": "loaded",
            "yield"     : "loaded",
            "cnn"       : "loaded" if CNN_MODEL is not None else "not loaded"
        },
        "cnn_classes": len(CNN_LABELS),
        "auth"       : "Firebase ID Token required"
    }), 200


@app.route("/register-listener", methods=["POST"])
@require_auth
def register_listener():
    uid = request.uid
    created = register_listener_for_user(uid)
    return jsonify({
        "message"     : "Listener registered for your account." if created else "Listener already active.",
        "listenerActive": True,
        "sensor_path" : f"/users/{uid}/sensors/latest",
        "config_path" : f"/users/{uid}/config",
        "predict_path": f"/users/{uid}/predictions",
        "cnn_path"    : f"/users/{uid}/cnn_prediction"
    }), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not TEST_MODE:
        listener_thread = threading.Thread(target=start_firebase_listener, daemon=True)
        listener_thread.start()
    else:
        log.info("Firebase listener disabled in TEST_MODE")
    
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
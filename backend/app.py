import os
import logging
import threading
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, jsonify, request

import firebase_admin
from firebase_admin import credentials, db

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Firebase init ─────────────────────────────────────────────────────────────
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": os.getenv("FIREBASE_DB_URL", "https://smart-agriculture-ai-6e8c5-default-rtdb.firebaseio.com")
})

# ── Load models ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

log.info("Loading models...")

disease_model     = joblib.load(os.path.join(BASE, "models", "crop_disease_model.pkl"))
disease_le        = joblib.load(os.path.join(BASE, "models", "disease_label_encoder.pkl"))

irrigation_model  = joblib.load(os.path.join(BASE, "models", "irrigation_model.pkl"))
irrigation_scaler = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
irrigation_le     = joblib.load(os.path.join(BASE, "models", "label_encoder.pkl"))

yield_model = joblib.load(os.path.join(BASE, "models", "crop_yield_model.pkl"))

log.info("All models loaded.")

# ── Crop name maps ────────────────────────────────────────────────────────────
# Key   = what user sets in Firebase config (always lowercase)
# Value = exact name the model was trained on

DISEASE_CROPS = {
    "rice"       : "rice",
    "maize"      : "maize",
    "chickpea"   : "chickpea",
    "kidneybeans": "kidneybeans",
    "pigeonpeas" : "pigeonpeas",
    "mothbeans"  : "mothbeans",
    "mungbean"   : "mungbean",
    "blackgram"  : "blackgram",
    "lentil"     : "lentil",
    "pomegranate": "pomegranate",
    "banana"     : "banana",
    "mango"      : "mango",
    "grapes"     : "grapes",
    "watermelon" : "watermelon",
    "muskmelon"  : "muskmelon",
    "apple"      : "apple",
    "orange"     : "orange",
    "papaya"     : "papaya",
    "coconut"    : "coconut",
    "cotton"     : "cotton",
    "jute"       : "jute",
    "coffee"     : "coffee"
}

IRRIGATION_CROPS = {
    "wheat"         : "Wheat",
    "maize"         : "Maize",
    "rice"          : "Paddy",
    "paddy"         : "Paddy",
    "potato"        : "Potato",
    "sugarcane"     : "Sugarcane",
    "coffee"        : "Coffee",
    "groundnuts"    : "Groundnuts",
    "pulse"         : "Pulse",
    "garden flowers": "Garden Flowers"
}

YIELD_CROPS = {
    "maize"       : "Maize",
    "wheat"       : "Wheat",
    "rice"        : "Rice, paddy",
    "sorghum"     : "Sorghum",
    "potato"      : "Potatoes",
    "sweet potato": "Sweet potatoes",
    "cassava"     : "Cassava",
    "soybean"     : "Soybeans",
    "yam"         : "Yams",
    "plantain"    : "Plantains"
}

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def predict_disease(sensor, config):
    crop = config.get("cropType", "").lower().strip()

    if crop not in DISEASE_CROPS:
        log.warning("Disease model — '%s' not supported, skipping.", crop)
        return {
            "label"      : "Not Available",
            "atRiskProb" : None,
            "healthyProb": None,
            "reason"     : f"'{crop}' is not supported by disease model.",
            "supported"  : sorted(list(DISEASE_CROPS.keys())),
            "timestamp"  : datetime.utcnow().isoformat()
        }

    crop_encoded = disease_le.transform([DISEASE_CROPS[crop]])[0]

    X = pd.DataFrame([{
        "N"           : float(sensor["N"]),
        "P"           : float(sensor["P"]),
        "K"           : float(sensor["K"]),
        "temperature" : float(sensor["temperature"]),
        "humidity"    : float(sensor["humidity"]),
        "ph"          : float(sensor["ph"]),
        "rainfall"    : float(sensor["rainfall"]),
        "crop_encoded": int(crop_encoded)
    }])

    prob  = disease_model.predict_proba(X)[0]
    label = disease_model.predict(X)[0]

    return {
        "label"      : "At_Risk" if label == 1 else "Healthy",
        "atRiskProb" : round(float(prob[1]), 4),
        "healthyProb": round(float(prob[0]), 4),
        "timestamp"  : datetime.utcnow().isoformat()
    }


def predict_irrigation(sensor, config):
    crop = config.get("cropType", "").lower().strip()

    if crop not in IRRIGATION_CROPS:
        log.warning("Irrigation model — '%s' not supported, skipping.", crop)
        return {
            "irrigate"  : None,
            "label"     : "Not Available",
            "confidence": None,
            "reason"    : f"'{crop}' is not supported by irrigation model.",
            "supported" : sorted(list(IRRIGATION_CROPS.keys())),
            "timestamp" : datetime.utcnow().isoformat()
        }

    crop_enc = irrigation_le.transform([IRRIGATION_CROPS[crop]])[0]

    X_raw = np.array([[
        int(crop_enc),
        int(config.get("cropDays", 30)),
        float(sensor["soilMoisture"]),
        float(sensor["temperature"]),
        float(sensor["humidity"])
    ]])

    X_scaled = irrigation_scaler.transform(X_raw)
    pred     = irrigation_model.predict(X_scaled)[0]
    prob     = irrigation_model.predict_proba(X_scaled)[0]

    return {
        "irrigate"  : int(pred),
        "label"     : "Irrigate" if pred == 1 else "No Irrigation",
        "confidence": round(float(max(prob)), 4),
        "timestamp" : datetime.utcnow().isoformat()
    }


def predict_yield(sensor, config):
    crop = config.get("cropType", "").lower().strip()

    if crop not in YIELD_CROPS:
        log.warning("Yield model — '%s' not supported, skipping.", crop)
        return {
            "hgPerHa"  : None,
            "kgPerHa"  : None,
            "reason"   : f"'{crop}' is not supported by yield model.",
            "supported": sorted(list(YIELD_CROPS.keys())),
            "timestamp": datetime.utcnow().isoformat()
        }

    X = pd.DataFrame([{
        "Area"                         : config.get("country", "India"),
        "Item"                         : YIELD_CROPS[crop],
        "Year"                         : int(config.get("year", datetime.utcnow().year)),
        "average_rain_fall_mm_per_year": float(sensor["rainfall"]),
        "pesticides_log"               : np.log1p(float(config.get("pesticides", 100))),
        "avg_temp"                     : float(sensor["temperature"])
    }])

    hg_per_ha = float(yield_model.predict(X)[0])

    return {
        "hgPerHa"  : round(hg_per_ha, 2),
        "kgPerHa"  : round(hg_per_ha / 10, 2),
        "timestamp": datetime.utcnow().isoformat()
    }


def run_all_predictions(sensor, config):
    results = {}
    crop    = config.get("cropType", "unknown").lower()

    log.info("Running predictions for crop: '%s'", crop)

    try:
        results["disease"] = predict_disease(sensor, config)
        log.info("Disease    → %s", results["disease"]["label"])
    except Exception as e:
        log.error("Disease prediction error: %s", e)
        results["disease"] = {"label": "Error", "error": str(e)}

    try:
        results["irrigation"] = predict_irrigation(sensor, config)
        log.info("Irrigation → %s", results["irrigation"]["label"])
    except Exception as e:
        log.error("Irrigation prediction error: %s", e)
        results["irrigation"] = {"label": "Error", "error": str(e)}

    try:
        results["yield"] = predict_yield(sensor, config)
        log.info("Yield      → %s kg/ha", results["yield"].get("kgPerHa", "N/A"))
    except Exception as e:
        log.error("Yield prediction error: %s", e)
        results["yield"] = {"label": "Error", "error": str(e)}

    log.info(
        "Complete — disease:%s  irrigation:%s  yield:%s",
        results["disease"]["label"],
        results["irrigation"]["label"],
        f"{results['yield'].get('kgPerHa', 'N/A')} kg/ha"
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FIREBASE LISTENER
# ─────────────────────────────────────────────────────────────────────────────

def on_sensor_update(event):
    if event.data is None:
        return

    log.info("New sensor data received from Firebase.")

    try:
        sensor = event.data
        config = db.reference("/config").get() or {}

        if not config:
            log.warning("No config found at /config — skipping prediction.")
            return

        predictions = run_all_predictions(sensor, config)
        db.reference("/predictions").set(predictions)
        log.info("Predictions written to /predictions")

    except Exception as e:
        log.error("Listener error: %s", e)


def start_firebase_listener():
    db.reference("/sensors/latest").listen(on_sensor_update)


# ─────────────────────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    body   = request.get_json(force=True)
    sensor = body.get("sensor", {})
    config = body.get("config", {})

    if not sensor or not config:
        return jsonify({"error": "Missing sensor or config in request body"}), 400

    results = run_all_predictions(sensor, config)
    db.reference("/predictions").set(results)

    return jsonify(results), 200


@app.route("/predict/now", methods=["GET"])
def predict_now():
    sensor = db.reference("/sensors/latest").get()
    config = db.reference("/config").get()

    if not sensor:
        return jsonify({"error": "No sensor data at /sensors/latest"}), 404
    if not config:
        return jsonify({"error": "No config at /config"}), 404

    results = run_all_predictions(sensor, config)
    db.reference("/predictions").set(results)

    return jsonify(results), 200


@app.route("/crops", methods=["GET"])
def supported_crops():
    """Returns all supported crops per model."""
    return jsonify({
        "disease_model"   : sorted(list(DISEASE_CROPS.keys())),
        "irrigation_model": sorted(list(IRRIGATION_CROPS.keys())),
        "yield_model"     : sorted(list(YIELD_CROPS.keys())),
        "works_in_all_3"  : sorted(list(
            set(DISEASE_CROPS) & set(IRRIGATION_CROPS) & set(YIELD_CROPS)
        ))
    }), 200


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status" : "running",
        "models" : ["disease", "irrigation", "yield"],
        "firebase": db.reference("/").get(shallow=True) is not None
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    listener_thread = threading.Thread(target=start_firebase_listener, daemon=True)
    listener_thread.start()
    log.info("Firebase listener started — watching /sensors/latest")

    app.run(host="0.0.0.0", port=5000, debug=False)
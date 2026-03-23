import os
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

# ── Firebase init ─────────────────────────────────────────────────────────────
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": os.getenv("FIREBASE_DB_URL", "https://smart-agriculture-ai-6e8c5-default-rtdb.firebaseio.com")
})

# ── Load models ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

log.info("Loading models...")

disease_model = joblib.load(os.path.join(BASE, "models", "crop_disease_model.pkl"))
disease_le    = joblib.load(os.path.join(BASE, "models", "disease_label_encoder.pkl"))
disease_name_model = joblib.load(os.path.join(BASE, "models", "crop_disease_name_model.pkl"))
disease_name_le    = joblib.load(os.path.join(BASE, "models", "disease_name_label_encoder.pkl"))
crop_name_le       = joblib.load(os.path.join(BASE, "models", "crop_name_encoder.pkl"))

irrigation_model  = joblib.load(os.path.join(BASE, "models", "irrigation_model.pkl"))
irrigation_scaler = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
irrigation_le     = joblib.load(os.path.join(BASE, "models", "label_encoder.pkl"))

yield_model = joblib.load(os.path.join(BASE, "models", "crop_yield_model.pkl"))

log.info("All models loaded.")

# ── Crop name maps ────────────────────────────────────────────────────────────
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
    "banana"        : "Bananas",
    "barley"        : "Barley",
    "bean"          : "Beans",
    "cassava"       : "Cassava",
    "coffee"        : "Coffee",
    "cotton"        : "Cotton",
    "groundnut"     : "Groundnuts",
    "maize"         : "Maize",
    "orange"        : "Oranges",
    "palm oil"      : "Palm oil",
    "pea"           : "Peas",
    "plantain"      : "Plantains and others",
    "potato"        : "Potatoes",
    "rapeseed"      : "Rapeseed",
    "rice"          : "Rice",
    "sorghum"       : "Sorghum",
    "soybean"       : "Soybeans",
    "sugarbeet"     : "Sugarbeet",
    "sugarcane"     : "Sugarcane",
    "sweet potato"  : "Sweet potatoes",
    "tomato"        : "Tomatoes",
    "wheat"         : "Wheat",
    "yam"           : "Yams"
}

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH DECORATOR
# Every protected endpoint must receive Authorization: Bearer <firebase_id_token>
# Flask verifies the token with Firebase Admin SDK — no secrets needed client side
# ─────────────────────────────────────────────────────────────────────────────

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        id_token = auth_header.split("Bearer ")[1]
        try:
            decoded = auth.verify_id_token(id_token)
            request.uid = decoded["uid"]
            request.user_email = decoded.get("email", "unknown")
        except Exception as e:
            log.warning("Token verification failed: %s", e)
            return jsonify({"error": "Invalid or expired token"}), 401
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────────────
# USER-SCOPED FIREBASE HELPERS
# All data is stored under /users/{uid}/ so each farmer is fully isolated
# ─────────────────────────────────────────────────────────────────────────────

def user_sensors_ref(uid):
    return db.reference(f"/users/{uid}/sensors/latest")

def user_predictions_ref(uid):
    return db.reference(f"/users/{uid}/predictions")

def user_config_ref(uid):
    return db.reference(f"/users/{uid}/config")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS  (unchanged logic, same as before)
# ─────────────────────────────────────────────────────────────────────────────

def predict_disease(sensor, config):
    crop = config.get("cropType", "").lower().strip()

    if crop not in DISEASE_CROPS:
        return {
            "label"      : "Not Available",
            "atRiskProb" : None,
            "healthyProb": None,
            "reason"     : f"'{crop}' is not supported by disease model.",
            "supported"  : sorted(list(DISEASE_CROPS.keys())),
            "timestamp"  : datetime.utcnow().isoformat()
        }

    crop_encoded = crop_name_le.transform([DISEASE_CROPS[crop]])[0]

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
    result = {
        "label"      : "At_Risk" if label == 1 else "Healthy",
        "atRiskProb" : round(float(prob[1]), 4),
        "healthyProb": round(float(prob[0]), 4),
        "diseaseName": None,
        "timestamp"  : datetime.utcnow().isoformat()
    }

    # If at risk — predict the specific disease name
    if label == 1:
        try:
            dn_raw = disease_name_le.inverse_transform(
                [disease_name_model.predict(X)[0]]
            )[0]
            result["diseaseName"] = dn_raw.split("__", 1)[-1]
        except Exception as e:
            log.warning("Disease name prediction failed: %s", e)
            result["diseaseName"] = "Unknown"

    return result

def predict_irrigation(sensor, config):
    crop = config.get("cropType", "").lower().strip()
    if crop not in IRRIGATION_CROPS:
        return {
            "irrigate"  : None,
            "label"     : "Not Available",
            "confidence": None,
            "reason"    : f"'{crop}' is not supported by irrigation model.",
            "supported" : sorted(list(IRRIGATION_CROPS.keys())),
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
    pred     = irrigation_model.predict(X_scaled)[0]
    prob     = irrigation_model.predict_proba(X_scaled)[0]

    # Rule override — soil moisture dominates over model bias
    soil = float(sensor["soilMoisture"])
    if soil >= 550:
        return {
            "irrigate"  : 1,
            "label"     : "Irrigate",
            "confidence": round(float(max(prob)), 4),
            "timestamp" : datetime.utcnow().isoformat()
        }
    elif soil <= 400:
        return {
            "irrigate"  : 0,
            "label"     : "No Irrigation",
            "confidence": round(float(max(prob)), 4),
            "timestamp" : datetime.utcnow().isoformat()
        }

    # Borderline zone (400-550) — trust the model
    return {
        "irrigate"  : int(pred),
        "label"     : "Irrigate" if pred == 1 else "No Irrigation",
        "confidence": round(float(max(prob)), 4),
        "timestamp" : datetime.utcnow().isoformat()
    }

def predict_yield(sensor, config):
    crop = config.get("cropType", "").lower().strip()
    if crop not in YIELD_CROPS:
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
        "average_rain_fall_mm_per_year": max(1.0, float(sensor["rainfall"])),
        "pesticides_tonnes"            : float(config.get("pesticides", 100)),
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

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FIREBASE LISTENER  — now watches ALL users under /users/
# When any user's ESP32 pushes sensor data, predictions run for that user only
# ─────────────────────────────────────────────────────────────────────────────

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
            user_predictions_ref(uid).set(predictions)
            log.info("Predictions written for uid=%s", uid)
        except Exception as e:
            log.error("Listener error for uid=%s: %s", uid, e)
    return handler


def register_listener_for_user(uid):
    user_sensors_ref(uid).listen(on_user_sensor_update(uid))
    log.info("Firebase listener registered for uid=%s", uid)


def start_firebase_listener():
    """
    Watch /users/ for any new child (new user registration).
    When a user node appears, attach a sensor listener for that user.
    """
    def on_new_user(event):
        if event.data is None:
            return
        # event.path is like '/abc123uid'
        uid = event.path.strip("/").split("/")[0]
        if uid:
            register_listener_for_user(uid)

    db.reference("/users").listen(on_new_user)
    log.info("Master listener started — watching /users/ for new farmers.")


# ─────────────────────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
@require_auth
def predict_endpoint():
    """
    POST /predict
    Header: Authorization: Bearer <firebase_id_token>
    Body:   { "sensor": {...}, "config": {...} }

    Runs predictions and writes results to /users/{uid}/predictions
    """
    body   = request.get_json(force=True)
    sensor = body.get("sensor", {})
    config = body.get("config", {})

    if not sensor or not config:
        return jsonify({"error": "Missing sensor or config in request body"}), 400

    results = run_all_predictions(sensor, config)
    user_predictions_ref(request.uid).set(results)

    log.info("Manual predict called by %s", request.user_email)
    return jsonify(results), 200


@app.route("/predict/now", methods=["GET"])
@require_auth
def predict_now():
    """
    GET /predict/now
    Header: Authorization: Bearer <firebase_id_token>

    Reads sensor data and config from /users/{uid}/... and runs predictions
    """
    uid    = request.uid
    sensor = user_sensors_ref(uid).get()
    config = user_config_ref(uid).get()

    if not sensor:
        return jsonify({"error": f"No sensor data at /users/{uid}/sensors/latest"}), 404
    if not config:
        return jsonify({"error": f"No config at /users/{uid}/config"}), 404

    results = run_all_predictions(sensor, config)
    user_predictions_ref(uid).set(results)

    log.info("predict/now called by %s", request.user_email)
    return jsonify(results), 200


@app.route("/crops", methods=["GET"])
def supported_crops():
    """Public endpoint — no auth needed, just lists supported crops"""
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
    """Public endpoint — health check"""
    return jsonify({
        "status" : "running",
        "models" : ["disease", "irrigation", "yield"],
        "firebase": db.reference("/").get(shallow=True) is not None,
        "auth"   : "Firebase ID Token required on /predict endpoints"
    }), 200


@app.route("/register-listener", methods=["POST"])
@require_auth
def register_listener():
    """
    POST /register-listener
    Header: Authorization: Bearer <firebase_id_token>

    Call this once after login to register the sensor listener for this user.
    The ESP32 should write to /users/{uid}/sensors/latest
    """
    uid = request.uid
    register_listener_for_user(uid)
    return jsonify({
        "message"     : f"Listener registered for your account.",
        "sensor_path" : f"/users/{uid}/sensors/latest",
        "config_path" : f"/users/{uid}/config",
        "predict_path": f"/users/{uid}/predictions"
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    listener_thread = threading.Thread(target=start_firebase_listener, daemon=True)
    listener_thread.start()

    app.run(host="0.0.0.0", port=5000, debug=False)
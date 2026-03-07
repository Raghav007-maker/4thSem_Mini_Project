from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ==============================
# LOAD TRAINED MODELS
# ==============================

crop_yield_model = joblib.load("models/crop_yield_model.pkl")
irrigation_model = joblib.load("models/irrigation_model.pkl")
crop_recommendation_model = joblib.load("models/crop_disease_model.pkl")

print("Models loaded successfully!")

# ==============================
# HOME ROUTE
# ==============================

@app.route("/")
def home():
    return "Smart Irrigation AI API is running"

# ==============================
# 1️⃣ CROP YIELD PREDICTION
# ==============================

@app.route("/predict_yield", methods=["POST"])
def predict_yield():

    try:

        data = request.json

        input_data = pd.DataFrame([{
            "Area": data["Area"],
            "Item": data["Item"],
            "Year": data["Year"],
            "average_rain_fall_mm_per_year": data["rainfall"],
            "pesticides_tonnes": data["pesticides"],
            "avg_temp": data["temperature"]
        }])

        prediction = crop_yield_model.predict(input_data)

        return jsonify({
            "predicted_yield": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# 2️⃣ IRRIGATION PREDICTION
# ==============================

@app.route("/predict_irrigation", methods=["POST"])
def predict_irrigation():

    try:

        data = request.json

        input_data = pd.DataFrame([{
            "soil_moisture": data["soil_moisture"],
            "temperature": data["temperature"],
            "humidity": data["humidity"]
        }])

        prediction = irrigation_model.predict(input_data)

        return jsonify({
            "irrigation_needed": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# 3️⃣ CROP RECOMMENDATION
# ==============================

@app.route("/recommend_crop", methods=["POST"])
def recommend_crop():

    try:

        data = request.json

        input_data = pd.DataFrame([{
            "N": data["nitrogen"],
            "P": data["phosphorus"],
            "K": data["potassium"],
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "ph": data["ph"],
            "rainfall": data["rainfall"]
        }])

        prediction = crop_recommendation_model.predict(input_data)

        return jsonify({
            "recommended_crop": prediction[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
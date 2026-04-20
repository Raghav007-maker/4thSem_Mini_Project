# 🌾 KISAAN AI — Precision Farming Intelligence Platform

> **4th Semester Mini Project** | B.Tech CSE | 2024–25
> Raghav Bansal · Anuj Kumar · Priya Shukla

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com)
[![Firebase](https://img.shields.io/badge/Firebase-Realtime_DB-orange.svg)](https://firebase.google.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

KISAAN AI is an IoT-powered smart agriculture platform that combines **real-time ESP32 sensor data** with **four AI models** to help Indian farmers make data-driven decisions about crop health, irrigation, and yield — all through a live web dashboard.

The system is designed for the Indian agricultural context, covering **48 crops** with thresholds sourced from ICAR (Indian Council of Agricultural Research) agronomic profiles.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🌡️ **Live Sensor Dashboard** | Real-time NPK, pH, soil moisture, temperature, humidity with sparkline charts |
| 🦠 **Crop Health Risk Engine** | Rule-based engine using ICAR thresholds — 48 crops, explainable risk scores |
| 💧 **Irrigation Recommender** | GradientBoosting model + soil moisture hard rules — 25 crops |
| 📈 **Yield Predictor** | RandomForest on FAOSTAT data — 23 crops, kg/ha output |
| 🍃 **CNN Leaf Disease Scanner** | MobileNetV2 fine-tuned on PlantVillage + 5 Indian crops, 99.7% val accuracy |
| 🔐 **Multi-user Auth** | Google OAuth + Email/Password via Firebase Authentication |
| 📱 **Mobile Responsive** | Full dashboard on phone/tablet/desktop |
| 🎭 **Demo Mode** | 4 cycling scenarios — works without ESP32 hardware |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ESP32 Device                            │
│   DHT22 (temp/humidity) · Capacitive Soil · pH · NPK sensors   │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Firebase Realtime DB
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              /users/{uid}/sensors/latest                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Firebase listener (auto-triggers)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Backend (app.py)                        │
│                                                                  │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ Disease Risk │  │  Irrigation   │  │   Yield Predictor    │  │
│  │   Engine     │  │  Recommender  │  │  (RandomForest)      │  │
│  │ (Rule-based) │  │  (GBM + rules)│  │  FAOSTAT pipeline    │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │          CNN Leaf Scanner  (MobileNetV2, 23 classes)     │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Writes predictions to Firebase
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              /users/{uid}/predictions                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Firebase real-time listener
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   dashboard.html                                 │
│     Vanilla JS · Firebase SDK 9.23.0 · Chart.js                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
4thSem_Mini_Project/
│
├── backend/
│   ├── app.py                        # Flask API + all prediction logic
│   ├── firebase-key.json             # 🔒 gitignored — service account key
│   └── models/
│       ├── crop_disease_model.pkl    # XGBoost (kept as fallback)
│       ├── disease_label_encoder.pkl
│       ├── crop_name_encoder.pkl
│       ├── irrigation_model.pkl      # GradientBoosting (25 crops)
│       ├── scaler.pkl                # StandardScaler for irrigation
│       ├── label_encoder.pkl         # Crop encoder for irrigation
│       ├── crop_yield_model.pkl      # RandomForest Pipeline (23 crops)
│       ├── plant_disease_cnn.h5      # MobileNetV2, 23 classes, 99.7% acc
│       └── cnn_class_labels.json     # {idx: {crop, disease, treatment}}
│
├── esp32/
│   ├── sensor_upload.ino             # ESP32 firmware (Arduino)
│   ├── test_all_models.py            # 50-scenario test suite
│   └── test_new_scenarios.py         # 33-scenario unseen test suite
│
├── notebooks/
│   ├── crop_recommendation_v3.ipynb  # Disease model training
│   ├── irrigation_final_pipeline_v2.ipynb
│   ├── crop_yield_fixed.ipynb
│   ├── plant_disease_cnn.ipynb       # PlantVillage CNN training
│   └── indian_crops_finetune.ipynb   # Fine-tuning for Indian crops
│
├── frontend/
│   ├── dashboard.html                # Full dashboard (single file)
│   └── firebase-config.js            # 🔒 gitignored — Firebase web config
│
└── README.md                         # This file
```

---

## 🤖 AI Models

### Model 1 — Crop Health Risk Engine
> Replaces XGBoost black-box with a transparent, explainable rule engine

- **Approach:** ICAR-sourced agronomic profiles for 48 crops
- **Inputs:** N, P, K, temperature, humidity, rainfall, pH, crop type
- **Output:** `Healthy` / `At_Risk` label + risk score (0–100) + plain-English reasons
- **Crops:** 48 (all major Indian crops including tropical crops like coconut, jute, papaya)
- **Key design:** Crop-specific thresholds — coconut's "dangerous" humidity (99%) is different from wheat's (78%)
- **Test accuracy:** 50/50 on original suite, 31/33 on unseen scenarios

**Example output:**
```json
{
  "label": "At_Risk",
  "riskScore": 72,
  "riskReasons": [
    "Humidity 91% dangerously high (danger threshold: 90%)",
    "Rainfall 280mm above ideal range (150–250mm)",
    "N=12 kg/ha critically low — nutrient deficiency weakens plant immunity",
    "Multiple simultaneous stress factors — compound risk elevated"
  ]
}
```

---

### Model 2 — Irrigation Recommender
- **Algorithm:** GradientBoostingClassifier
- **Features:** CropType, CropDays, SoilMoisture, Temperature, Humidity
- **Crops:** 25 Indian crops
- **Soil moisture scale:** 120 = wet, 800 = dry (capacitive sensor, inverted)
- **Hard rules override model:**
  - `soil ≥ 550` → **Irrigate** (dry)
  - `soil ≤ 420` → **No Irrigation** (wet)
  - `420 < soil < 550` → Trust GBM model
- **Test accuracy:** 50/50 on original suite

---

### Model 3 — Yield Predictor
- **Algorithm:** RandomForestRegressor with OneHotEncoder Pipeline
- **Data source:** FAOSTAT via Our World in Data
- **Features:** Country, Crop (Item), Year, Annual Rainfall, Pesticides, Avg Temperature
- **Crops:** 23 internationally tracked crops
- **Output:** kg/ha estimate
- **Sanity check:** Flags predictions outside 500–1,000,000 hg/ha

---

### Model 4 — CNN Leaf Disease Scanner
- **Architecture:** MobileNetV2 (Transfer Learning, ImageNet weights)
- **Base dataset:** PlantVillage (54,000+ images, 38 classes)
- **Fine-tuned for Indian crops:**

| Crop | Dataset |
|------|---------|
| Rice | `nirmalsankalana/rice-leaf-disease-image` |
| Wheat | `olyadgetch/wheat-leaf-dataset` |
| Cotton | `seroshkarim/cotton-leaf-disease-dataset` |
| Banana | `shifatearman/bananalsd` |
| Mango | `warcoder/mango-leaf-disease-dataset` |

- **Final model:** 23 disease classes across 19 crops
- **Validation accuracy:** 99.7% | Top-3 accuracy: 99.9%
- **Input:** 224×224 RGB image (base64 via API)
- **Output:** Top-3 predictions with crop, disease name, confidence, treatment advice

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js (for Live Server extension in VS Code)
- Firebase project with Realtime Database enabled
- Google Cloud project with OAuth 2.0 configured

### 1. Clone the Repository
```bash
git clone https://github.com/Raghav007-maker/4thSem_Mini_Project.git
cd 4thSem_Mini_Project
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install flask flask-cors firebase-admin joblib pandas numpy \
            scikit-learn xgboost tensorflow pillow
```

### 3. Firebase Configuration
Place your Firebase service account key at:
```
backend/firebase-key.json
```
> Download from Firebase Console → Project Settings → Service Accounts → Generate New Private Key

### 4. Frontend Configuration
Create `frontend/firebase-config.js`:
```javascript
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  databaseURL: "https://YOUR_PROJECT-default-rtdb.firebaseio.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};
```

### 5. Run the Backend
```bash
cd backend
python app.py
```

Expected startup output:
```
09:00:00  INFO  Loading sklearn models...
09:00:02  INFO  Sklearn models loaded.
09:00:02  INFO  Master listener started — watching /users/ for new farmers.
 * Running on http://0.0.0.0:5000
09:00:07  INFO  CNN model loaded — 23 classes.
```

### 6. Run the Frontend
Open `frontend/dashboard.html` with VS Code Live Server on port 8080.

---

## 🔌 API Reference

All prediction endpoints require Firebase ID Token in the `Authorization` header:
```
Authorization: Bearer <firebase_id_token>
```

### POST `/predict`
Manual prediction from sensor + config payload.
```json
{
  "sensor": {
    "temperature": 28, "humidity": 75, "soilMoisture": 350,
    "rainfall": 180, "ph": 6.5, "N": 80, "P": 50, "K": 55
  },
  "config": {
    "cropType": "rice", "country": "India",
    "cropDays": 60, "pesticides": 300, "year": 2024
  }
}
```

### GET `/predict/now`
Reads latest sensor data from Firebase and runs all models.

### POST `/predict/image`
CNN leaf disease scanner.
```json
{ "image": "<base64_encoded_image>" }
```

### GET `/crops`
Lists all supported crops per model.

### GET `/status`
Health check — shows model load status and Firebase connectivity.

---

## 🧪 Testing

### Run Original 50-Scenario Suite
```bash
# Ensure Flask is running first
cd esp32
python test_all_models.py
# Duration: ~10 minutes (50 scenarios × 12s)
```

### Run New Unseen 33-Scenario Suite
```bash
cd esp32
python test_new_scenarios.py
# Duration: ~7 minutes (33 scenarios × 12s)
```

### Test Suite Results

| Model | Original (50) | Unseen (33) |
|-------|:---:|:---:|
| Disease Risk Engine | 50/50 ✅ | 31/33 ✅ |
| Irrigation Recommender | 50/50 ✅ | 33/33 ✅ |
| Yield Predictor | 50/50 ✅ | — |
| **Overall** | **150/150** | **97%+** |

---

## 📊 ProjectFlow Score (March 2026)

| Category | Score | Grade |
|----------|:-----:|-------|
| Security | 73/100 | Good |
| Originality | 83/100 | Excellent |
| Documentation | 15/100 | Poor → improving |
| **Total** | **46/100** | Fair |

> Documentation score is being actively improved with this README and inline code comments.

---

## 🌐 Firebase Database Structure

```
/users/
  {uid}/
    config/
      cropType: "rice"
      country: "India"
      cropDays: 60
      pesticides: 300
      year: 2024
    sensors/
      latest/
        temperature: 28
        humidity: 75
        soilMoisture: 350
        rainfall: 180
        ph: 6.5
        N: 80
        P: 50
        K: 45
        timestamp: "2024-..."
    predictions/
      disease/
        label: "Healthy"
        riskScore: 12
        riskReasons: [...]
        atRiskProb: 0.12
        timestamp: "..."
      irrigation/
        label: "No Irrigation"
        confidence: 0.91
        timestamp: "..."
      yield/
        kgPerHa: 4301
        hgPerHa: 43010
        timestamp: "..."
    cnn_prediction/
      crop: "Rice"
      disease: "Leaf Blast"
      is_healthy: false
      confidence: 94.2
      treatment: "Apply tricyclazole..."
      top3: [...]
      timestamp: "..."
```

---

## 🔧 ESP32 Sensor Configuration

| Sensor | Model | Pin | Notes |
|--------|-------|-----|-------|
| Temperature + Humidity | DHT22 | GPIO 4 | ±0.5°C accuracy |
| Soil Moisture | Capacitive v1.2 | ADC GPIO 34 | Scale: 120=wet, 800=dry |
| pH | Analog pH module | ADC GPIO 35 | Calibrate with buffer solutions |
| NPK | RS485 NPK sensor | UART GPIO 16/17 | Modbus RTU protocol |

---

## ⚠️ Known Limitations

1. **Yield model magnitude** — FAOSTAT data is country-level average, not field-level. Predictions are indicative, not precise.
2. **CNN scanner** — requires clear, well-lit leaf photos. Background clutter reduces accuracy.
3. **Irrigation model** — the 420–550 soil moisture middle zone relies on the GBM model which may be less reliable for crops added after v1 (onion, sunflower, ginger, turmeric).
4. **No offline mode** — requires internet connection for Firebase sync.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| IoT Hardware | ESP32, DHT22, Capacitive Soil Sensor, pH Module, NPK Sensor |
| Frontend | Vanilla JavaScript, Firebase SDK 9.23.0 (compat), Chart.js |
| Backend | Python 3.10, Flask 3.x, flask-cors, firebase-admin |
| ML — Classical | scikit-learn, XGBoost, GradientBoosting, RandomForest |
| ML — Deep Learning | TensorFlow 2.x, Keras, MobileNetV2 |
| Database | Firebase Realtime Database |
| Authentication | Firebase Authentication (Google OAuth + Email/Password) |
| Data Sources | FAOSTAT / Our World in Data, PlantVillage, Kaggle crop datasets |
| Agronomic Reference | ICAR crop production guidelines |

---

## 👥 Team

| Name | Role |
|------|------|
| Raghav Bansal | Backend, ML models, Firebase integration, ESP32 firmware |
| Anuj Kumar | Frontend dashboard, UI/UX, authentication flow |
| Priya Shukla | CNN model training, dataset preparation, testing |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

> **Disclaimer:** The ML models in this project are for educational demonstration purposes. Do not use predictions for actual farming decisions without consulting a qualified agronomist.

---

## 🙏 Acknowledgements

- [ICAR](https://icar.org.in) — Agronomic thresholds and crop production guidelines
- [PlantVillage](https://plantvillage.psu.edu) — Leaf disease image dataset
- [Our World in Data / FAOSTAT](https://ourworldindata.org/crop-yields) — Yield data
- [Firebase](https://firebase.google.com) — Realtime database and authentication
- [TensorFlow](https://tensorflow.org) — MobileNetV2 transfer learning
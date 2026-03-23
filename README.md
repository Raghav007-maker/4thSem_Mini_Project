# 🌾 KISAAN AI — Precision Farming Intelligence Platform

> AI Smart Agriculture · 4th Semester Mini Project · Team 57

## 📖 Overview

KISAAN AI is a complete end-to-end precision farming platform that connects IoT sensors (ESP32) to a cloud-based AI backend. It collects real-time soil and environmental data, runs it through three machine learning models, and displays live predictions on a modern web dashboard — helping farmers make data-driven decisions about disease risk, irrigation, and expected crop yield.

**KISAAN** means *farmer* in Hindi — the name reflects the project's purpose of empowering Indian farmers with AI.

---

## 🏗️ System Architecture

```
ESP32 Sensors
    │
    │  writes every 30 seconds
    ▼
Firebase Realtime DB
/users/{uid}/sensors/latest
    │
    │  Firebase listener (app.py)
    ▼
Flask ML Backend
    │
    ├──► Disease Model      → Healthy / At_Risk + disease name
    ├──► Irrigation Model   → Irrigate / No Irrigation + confidence
    └──► Yield Model        → kg/ha + hg/ha forecast
    │
    ▼
Firebase Realtime DB
/users/{uid}/predictions
    │
    ▼
Web Dashboard (dashboard.html)
Real-time display — isolated per farmer account
```

---

## 📁 Folder Structure

```
AI-Smart-Agriculture/
├── .gitignore
├── requirements.txt
├── README.md
│
├── esp32/
│   ├── esp32_sender.ino          ← Arduino code for ESP32 hardware
│   ├── test_sensor.py            ← Simulate sensors without hardware
│   └── test_all_models.py        ← Test all 3 models across 6 scenarios
│
├── backend/
│   ├── app.py                    ← Main Flask backend + Firebase listener
│   ├── firebase-key.json         ← Firebase service account (gitignored)
│   └── models/
│       ├── crop_disease_model.pkl
│       ├── disease_label_encoder.pkl
│       ├── crop_disease_name_model.pkl
│       ├── crop_name_encoder.pkl
│       ├── disease_name_label_encoder.pkl
│       ├── irrigation_model.pkl
│       ├── scaler.pkl
│       ├── label_encoder.pkl
│       └── crop_yield_model.pkl
│
├── data/
│   ├── raw/yield_df.csv
│   └── processed/
│
├── notebooks/
│   ├── crop_recommendation.ipynb       ← Disease model Phase 1
│   ├── crop_recommendation_v2.ipynb    ← Disease model Phase 2 (active)
│   ├── irrigation_final_pipeline.ipynb ← Irrigation model
│   └── crop_yield_fixed.ipynb          ← Yield model
│
└── frontend/
    ├── dashboard.html                  ← Main web dashboard
    ├── firebase-config.js              ← Firebase credentials (gitignored)
    └── firebase-config.example.js     ← Template — copy and fill in values
```

---

## 🤖 Machine Learning Models

### Model 1 — Disease Risk Detection

| Property | Value |
|---|---|
| Algorithm | RandomForestClassifier |
| Task | Binary Classification — Healthy vs At_Risk |
| Accuracy | 84.6% (Phase 1) |
| ROC-AUC | 0.852 |
| Features | N, P, K, temperature, humidity, ph, rainfall, crop_encoded |
| Output | Healthy / At_Risk + probability + disease name |
| Supported Crops | 49 crops (rice, maize, wheat, banana, mango, coffee, apple, grapes, and more) |
| Notebook | `notebooks/crop_recommendation_v2.ipynb` |

**Disease Name Prediction** — when At_Risk is detected the model also predicts the specific disease name (e.g. Magnaporthe oryzae, Puccinia sorghi) using a second RandomForest trained on crop-scoped labels (`crop__disease_name` format).

---

### Model 2 — Smart Irrigation

| Property | Value |
|---|---|
| Algorithm | GradientBoostingClassifier |
| Task | Binary Classification — Irrigate vs No Irrigation |
| Accuracy | 92.4% test, 91.5% CV |
| ROC-AUC | 0.9796 |
| Training Rows | 6,479 (augmented from 501 original) |
| Features | CropType_enc, CropDays, SoilMoisture, temperature, Humidity |
| Output | Irrigate / No Irrigation + confidence score |
| Supported Crops | Wheat, Maize, Paddy, Potato, Sugarcane, Coffee, Groundnuts, Pulse, Garden Flowers |
| Notebook | `notebooks/irrigation_final_pipeline.ipynb` |

**Rule Override** — soil moisture is the dominant feature. If `soilMoisture >= 550` the model forces Irrigate. If `soilMoisture <= 400` it forces No Irrigation. The ML model handles borderline cases (400–550).

> Soil moisture scale: 120–450 = wet, 500–800 = dry. ESP32 maps raw ADC using `map(raw, 4095, 0, 120, 800)`

---

### Model 3 — Crop Yield Forecast

| Property | Value |
|---|---|
| Algorithm | RandomForestRegressor in sklearn Pipeline with OneHotEncoder |
| Task | Regression — predict crop yield in hg/ha |
| Test R² | 0.9607 |
| Train R² | 0.9845 |
| MAE | 1,329 kg/ha |
| Training Rows | 32,208 (OWID API + Kaggle combined) |
| Features | Area (country), Item (crop), Year, rainfall, pesticides_tonnes, avg_temp |
| Output | hg/ha + kg/ha |
| Supported Crops | 23 crops (Bananas, Barley, Beans, Cassava, Coffee, Cotton, Groundnuts, Maize, Oranges, Palm oil, Peas, Potatoes, Rapeseed, Rice, Sorghum, Soybeans, Sugarbeet, Sugarcane, Sweet potatoes, Tomatoes, Wheat, Yams, Plantains) |
| Notebook | `notebooks/crop_yield_fixed.ipynb` |

---

## 🔥 Firebase Database Structure

```json
{
  "/users/{uid}/sensors/latest": {
    "temperature": 28.5,
    "humidity": 75.0,
    "soilMoisture": 450,
    "rainfall": 120.0,
    "ph": 6.5,
    "N": 50, "P": 40, "K": 35,
    "timestamp": "2024-01-01T12:00:00Z"
  },
  "/users/{uid}/config": {
    "cropType": "maize",
    "country": "India",
    "cropDays": 45,
    "pesticides": 500,
    "year": 2024
  },
  "/users/{uid}/predictions": {
    "disease": {
      "label": "At_Risk",
      "atRiskProb": 0.81,
      "healthyProb": 0.19,
      "diseaseName": "Magnaporthe_oryzae",
      "timestamp": "2024-01-01T12:00:05Z"
    },
    "irrigation": {
      "irrigate": 1,
      "label": "Irrigate",
      "confidence": 0.95,
      "timestamp": "2024-01-01T12:00:05Z"
    },
    "yield": {
      "hgPerHa": 45861.9,
      "kgPerHa": 4586.19,
      "timestamp": "2024-01-01T12:00:05Z"
    }
  },
  "/users/{uid}/location": {
    "lat": 28.98,
    "lon": 77.71,
    "updatedAt": "2024-01-01T12:00:00Z"
  },
  "/users/{uid}/profile": {
    "name": "Farmer Name",
    "email": "farmer@example.com",
    "photoURL": "...",
    "createdAt": "2024-01-01T00:00:00Z"
  }
}
```

Each farmer's data is fully isolated under their Firebase UID. Firebase security rules ensure a user can only read and write their own data.

---

## 🔧 Hardware — ESP32 Sensor Setup

| Sensor | Parameter | GPIO Pin | Notes |
|---|---|---|---|
| DHT22 | Temperature + Humidity | GPIO 4 | Direct, no conversion |
| Capacitive Soil Moisture | soilMoisture | GPIO 34 (Analog) | `map(raw, 4095, 0, 120, 800)` |
| pH Sensor | ph | GPIO 35 (Analog) | Calibrate with buffer solutions |
| Rain Gauge | rainfall | Replaced by Weather API | Open-Meteo API — no hardware needed |
| NPK Sensor | N, P, K | Removed | Using crop-based default values |

**Critical soil moisture mapping:**
```cpp
// Higher ADC = drier soil on capacitive sensor
int soilMoisture = map(soilRaw, 4095, 0, 120, 800);
// 120 = fully wet   → No Irrigation
// 800 = bone dry    → Irrigate immediately
```

---

## 🌧️ Weather API Integration

Rainfall data is fetched automatically from **Open-Meteo API** (free, no API key required) using the farmer's browser geolocation. No rain gauge hardware needed.

```
https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&forecast_days=1&timezone=auto
```

- Daily rainfall in mm is converted to annual mm/yr for model compatibility
- Location is saved to Firebase so repeat logins don't need to ask again
- Falls back to saved location if geolocation is denied
- Updates every 30 minutes automatically

---

## 🔒 Security

| Feature | Status |
|---|---|
| Firebase API key removed from source code | ✅ Done |
| API key stored in `firebase-config.js` (gitignored) | ✅ Done |
| API key restricted to `localhost` in Google Cloud Console | ✅ Done |
| Firebase Auth — Email/Password | ✅ Done |
| Firebase Auth — Google Sign-In | ✅ Done |
| Flask endpoints protected with Firebase ID token verification | ✅ Done |
| Per-user Firebase data isolation (`/users/{uid}/...`) | ✅ Done |
| Firebase security rules — users can only access own data | ✅ Done |
| `firebase-key.json` gitignored | ✅ Done |
| Model `.pkl` files gitignored | ✅ Done |

**Firebase Security Rules:**
```json
{
  "rules": {
    "users": {
      "$uid": {
        ".read":  "$uid === auth.uid",
        ".write": "$uid === auth.uid"
      }
    }
  }
}
```

---

## 🚀 Setup & Running

### Prerequisites

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
flask
flask-cors
firebase-admin
scikit-learn
pandas
numpy
joblib
```

### Step 1 — Firebase Setup

1. Create project at [console.firebase.google.com](https://console.firebase.google.com)
2. Enable Realtime Database and Email/Password + Google Authentication
3. Download service account key → save as `backend/firebase-key.json`
4. Apply security rules from `firebase-rules.json` in Firebase Console → Realtime Database → Rules

### Step 2 — Frontend Config

```bash
cp frontend/firebase-config.example.js frontend/firebase-config.js
# Edit firebase-config.js and fill in your Firebase credentials
```

### Step 3 — Generate ML Models

Run notebooks in this order:

```bash
jupyter notebook
```

1. `notebooks/irrigation_final_pipeline.ipynb` — run all cells
2. `notebooks/crop_yield_fixed.ipynb` — run all cells
3. `notebooks/crop_recommendation_v2.ipynb` — run all cells

All `.pkl` files save to `backend/models/` automatically.

### Step 4 — Start Flask Backend

```bash
cd backend
python app.py
```

Wait for:
```
All models loaded
Firebase listener started
```

### Step 5 — Run Dashboard

```bash
cd frontend
python -m http.server 8080
```

Open: **http://localhost:8080/dashboard.html**

### Step 6 — Test Without Hardware

```bash
cd esp32
python test_sensor.py
```

Or run the full model test suite:

```bash
cd esp32
python test_all_models.py
```

---

## 📊 Dashboard Features

- **Splash animation** — seed particle animation on first load
- **Login / Register** — Email/Password or Google Sign-In
- **Profile management** — upload photo, edit display name
- **Hero banner** — full crop-specific background image (changes per crop type)
- **8 sensor cards** — temperature, humidity, soil moisture, rainfall, pH, N, P, K with live sparkline charts
- **3 AI prediction cards** — Disease Risk (donut chart + disease name), Smart Irrigation (confidence bar), Yield Forecast (regional comparison)
- **Live weather widget** — today's rainfall from Open-Meteo, auto-detects location
- **Soil Quality Index** — calculated from pH, N, P, K values (0–100 score)
- **24h telemetry chart** — rolling moisture, temperature, humidity history
- **System Intelligence Feed** — real-time alerts from predictions
- **Bell notifications** — all events pushed to notification panel
- **Demo Mode** — cycles 4 realistic scenarios every 5 seconds
- **Settings panel** — update crop, country, days, pesticides, year
- **Per-user isolation** — each farmer sees only their own data

---

## 🗺️ Crop Image Support

Hero banner automatically shows a real field photograph for each crop:

| Crop | Image |
|---|---|
| Maize | Corn field |
| Rice | Paddy field |
| Wheat | Golden wheat field |
| Cotton | Cotton field |
| Coffee | Coffee plantation |
| Banana, Mango, Grapes, Apple, Orange | Fruit orchards |
| Potato, Tomato, Sugarcane, Coconut | Respective crop fields |
| All others | Default farm landscape |

---

## 🔗 API Endpoints

All endpoints except `/crops` and `/status` require `Authorization: Bearer <firebase_id_token>` header.

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/predict` | Required | Run predictions with custom sensor + config data |
| GET | `/predict/now` | Required | Run predictions using user's current Firebase data |
| POST | `/register-listener` | Required | Register real-time sensor listener for this user |
| GET | `/crops` | Public | List all supported crops per model |
| GET | `/status` | Public | Health check — models loaded, Firebase connected |

---

## 🗂️ Development Phases

| Phase | Task | Status |
|---|---|---|
| Phase 1 | Disease model (22 crops, 84.6%) | ✅ Complete |
| Phase 1 | Irrigation model (9 crops, 92.4%) | ✅ Complete |
| Phase 1 | Yield model | ✅ Complete |
| Phase 1 | Flask backend + Firebase integration | ✅ Complete |
| Phase 2 | Yield model (23 crops, R²=0.9607, OWID+Kaggle) | ✅ Complete |
| Phase 2 | Disease model v2 (49 crops + disease names) | ✅ Complete |
| Phase 2 | Multi-user authentication + user-scoped data | ✅ Complete |
| Phase 2 | Google Sign-In | ✅ Complete |
| Phase 2 | Weather API replacing rain gauge hardware | ✅ Complete |
| Phase 2 | Dashboard redesign (KISAAN AI) | ✅ Complete |
| Phase 2 | Security hardening (API key, Firebase rules) | ✅ Complete |
| Phase 2 | Disease name prediction | ✅ Complete |
| Phase 2 | Irrigation rule override (soil moisture dominant) | ✅ Complete |
| Phase 2 | Irrigation model retrain (FAO AQUASTAT data) | ⏳ Pending |

---

## ⚠️ Known Issues

| Issue | Details | Status |
|---|---|---|
| Irrigation humidity bias | Model overweights humidity vs soil moisture | Fixed with rule override |
| Disease model synthetic labels | Labels generated from crop profiles, not real observations | Acceptable for 4th sem project |
| Yield climate data limited | OWID yields merged with Kaggle — only 101 countries match | Accepted tradeoff |
| Rice duplicate in OWID | OWID has 'Rice', Kaggle has 'Rice, paddy' — merged as 'Rice' | Fixed in notebook |
| CV memory crash on Windows | `cross_val_score` with `n_jobs=-1` kills process | Fixed with `n_jobs=1` |
| Google Auth popup blocked | Browser may block popup — allow popups for localhost | User setting |

---

## 👥 Team

| Name | GitHub | Contribution |
|---|---|---|
| Raghav Bansal | [@Raghav007-maker](https://github.com/Raghav007-maker) | Backend, ML models, Firebase, Security |

| Priya Shukla | [@PriyaShukla3694](https://github.com/PriyaShukla3694) | Frontend dashboard, UI/UX |

| Anuj Kumar | [@aksaxena9412-ctrl] |Esp32 , Sensor codes , hardware integration  |

**GitHub:** [https://github.com/Raghav007-maker/4thSem_Mini_Project](https://github.com/Raghav007-maker/4thSem_Mini_Project)

**Firebase Project:** `smart-agriculture-ai-6e8c5`

---

## 📄 License

Academic project — 4th Semester Mini Project. Not licensed for commercial use.

---

*KISAAN AI · Built with Flask · Firebase · scikit-learn · ESP32 · Chart.js · Open-Meteo*
# Firebase Realtime Database — Expected Structure
# ─────────────────────────────────────────────────────────────
# ESP32 writes to   /sensors/latest
# User sets once    /config
# Flask writes to   /predictions
# Dashboard reads   /predictions
# ─────────────────────────────────────────────────────────────

{
  "sensors": {
    "latest": {
      "temperature"  : 28.5,      # °C         — from DHT22 / DS18B20
      "humidity"     : 75.0,      # %RH         — from DHT22
      "soilMoisture" : 450,       # raw/mapped  — from capacitive sensor
      "rainfall"     : 120.0,     # mm/year     — from rain gauge or manual
      "ph"           : 6.5,       # 0-14        — from pH sensor
      "N"            : 50,        # mg/kg       — from NPK sensor
      "P"            : 40,        # mg/kg       — from NPK sensor
      "K"            : 35,        # mg/kg       — from NPK sensor
      "timestamp"    : "2024-01-15T10:30:00"
    }
  },

  "config": {
    "cropType"   : "Wheat",       # must match training data labels exactly
    "country"    : "India",       # must match yield model Area labels
    "cropDays"   : 45,            # days since planting (updated by user)
    "pesticides" : 500,           # tonnes — for yield model
    "year"       : 2024           # current year — for yield model
  },

  "predictions": {
    "disease": {
      "label"      : "Healthy",   # "Healthy" or "At_Risk"
      "atRiskProb" : 0.23,        # probability of At_Risk (0-1)
      "healthyProb": 0.77,
      "timestamp"  : "2024-01-15T10:30:05"
    },
    "irrigation": {
      "irrigate"   : 0,           # 0 = No, 1 = Yes
      "label"      : "No Irrigation",
      "confidence" : 0.95,        # how confident the model is (0-1)
      "timestamp"  : "2024-01-15T10:30:05"
    },
    "yield": {
      "hgPerHa"  : 29175.0,       # hectograms per hectare
      "kgPerHa"  : 2917.5,        # kg per hectare (hgPerHa / 10)
      "timestamp": "2024-01-15T10:30:05"
    }
  }
}

# ─────────────────────────────────────────────────────────────
# Valid cropType values (must match exactly):
#   Wheat, Rice, Maize, Sugarcane, Potato, Paddy,
#   Coffee, Groundnuts, Garden Flowers, Pulse
#
# Valid country values (for yield model):
#   India, Brazil, Australia, Japan, Indonesia,
#   Mexico, Pakistan, and all other countries in dataset
# ─────────────────────────────────────────────────────────────

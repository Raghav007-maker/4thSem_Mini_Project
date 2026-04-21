import firebase_admin
from firebase_admin import credentials, db
import time
import os
from datetime import datetime, timezone

USER_UID = os.getenv("FIREBASE_TEST_UID")
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "../backend/firebase-key.json")

if not USER_UID:
    raise RuntimeError("Set FIREBASE_TEST_UID in your environment before running this script.")

if not FIREBASE_DB_URL:
    raise RuntimeError("Set FIREBASE_DB_URL in your environment before running this script.")

cred = credentials.Certificate(FIREBASE_KEY_PATH)
firebase_admin.initialize_app(cred, {
    "databaseURL": FIREBASE_DB_URL
})

# ── Fake sensor data — edit these values to test different scenarios ──────────
sensor_data = {
    "temperature"  : 28.5,
    "humidity"     : 75.0,
    "soilMoisture" : 450,
    "rainfall"     : 120.0,
    "ph"           : 6.5,
    "N"            : 60,
    "P"            : 55,
    "K"            : 80,
    "timestamp"    : datetime.now(timezone.utc).isoformat()
}

# ── Config — tells Flask which crop/country to use for predictions ────────────
config_data = {
    "cropType"   : "maize",
    "country"    : "India",
    "cropDays"   : 45,
    "pesticides" : 500,
    "year"       : 2024
}

print(f"Writing data for user: {USER_UID[:8]}...")
print(f"Sensor data: {sensor_data}")
print(f"Config data: {config_data}")

# Write config first so Flask has crop info when sensor triggers
db.reference(f"/users/{USER_UID}/config").set(config_data)
print("\nConfig written.")

# Write sensor data — this triggers Flask listener → predictions
db.reference(f"/users/{USER_UID}/sensors/latest").set(sensor_data)
print("Sensor data written.")

print("\nWaiting 5 seconds for Flask to process...")
time.sleep(5)

predictions = db.reference(f"/users/{USER_UID}/predictions").get()

if predictions:
    print("\n=== PREDICTIONS FROM FIREBASE ===")
    for model, result in predictions.items():
        print(f"\n{model.upper()}:")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {result}")
else:
    print("\nNo predictions found yet.")
    print("Make sure app.py is running and USER_UID is correct.")
import firebase_admin
from firebase_admin import credentials, db
import time
from datetime import datetime

# ── IMPORTANT: paste your Firebase UID here ──────────────────────────────────
# Get it from browser console after login:
#   firebase.auth().currentUser.uid
USER_UID = "D0H5vS9PTfXENKobo8s54qAluf52"
# ─────────────────────────────────────────────────────────────────────────────

cred = credentials.Certificate("../backend/firebase-key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-agriculture-ai-6e8c5-default-rtdb.firebaseio.com"
})

# ── Fake sensor data — edit these values to test different scenarios ──────────
sensor_data = {"temperature":30.0,"humidity":55.0,"soilMoisture":480,"rainfall":80.0,"ph":6.5,"N":65,"P":52,"K":75,"timestamp":datetime.utcnow().isoformat()}
config_data  = {"cropType":"rice","country":"India","cropDays":45,"pesticides":300,"year":2024}
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
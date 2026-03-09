import firebase_admin
from firebase_admin import credentials, db
import time
from datetime import datetime

cred = credentials.Certificate("../backend/firebase-key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://smart-agriculture-ai-6e8c5-default-rtdb.firebaseio.com"   # change this
})

# Fake sensor data — edit these values to test different scenarios
sensor_data = {
    "temperature"  : 42.0,
    "humidity"     : 88.0,
    "soilMoisture" : 250,
    "rainfall"     : 250.0,
    "ph"           : 4.5,
    "N": 20, "P": 15, "K": 10,
    "timestamp"    : datetime.utcnow().isoformat()
}

print("Writing sensor data to Firebase...")
print(sensor_data)

db.reference("/sensors/latest").set(sensor_data)

print("\nDone. Now check:")
print("  1. Your Flask terminal — should show prediction logs")
print("  2. Firebase console → /predictions — should have results")

# Wait and then read back the predictions
print("\nWaiting 5 seconds for Flask to process...")
time.sleep(5)

predictions = db.reference("/predictions").get()

if predictions:
    print("\n=== PREDICTIONS FROM FIREBASE ===")
    for model, result in predictions.items():
        print(f"\n{model.upper()}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
else:
    print("\nNo predictions found yet — make sure app.py is running.")
# 🔧 Recommended Code Fixes for KISAAN AI

This file contains code patches for the remaining high-priority security issues identified in [../status/SECURITY_AND_BUG_REPORT.md](../status/SECURITY_AND_BUG_REPORT.md).

---

## Fix 1: Add Rate Limiting (Issue #7)

Add this to `../../backend/app.py`:

```python
# At the top with other imports:
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# After creating the Flask app instance:
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",  # Use Redis for production: "redis://localhost:6379"
    default_limits=["200 per day", "50 per hour"]
)

# Then decorate endpoints:
@app.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")  # 10 predictions per minute per IP
@require_auth
def predict_endpoint():
    # ... existing code ...

@app.route("/predict/now", methods=["GET"])
@limiter.limit("10 per minute")
@require_auth
def predict_now():
    # ... existing code ...

@app.route("/predict/image", methods=["POST"])
@limiter.limit("5 per minute")  # Image processing is heavier
@require_auth
def predict_image():
    # ... existing code ...
```

**Install dependency:**
```bash
pip install Flask-Limiter redis
```

---

## Fix 2: Input Sanitization for Logging (Issue #8)

Replace these lines in `predict_disease()`, `predict_irrigation()`, and `predict_yield()`:

```python
# BEFORE (vulnerable to log injection):
log.info("Health risk [%s] → %s (score=%d)", crop, result["label"], result["riskScore"])

# AFTER (sanitized):
def sanitize_for_log(value):
    """Remove newlines and control characters to prevent log injection."""
    if not isinstance(value, str):
        return str(value)
    return value.replace("\n", "\\n").replace("\r", "\\r")[:100]

log.info("Health risk [%s] → %s (score=%d)", 
         sanitize_for_log(crop), 
         sanitize_for_log(result["label"]), 
         result["riskScore"])
```

---

## Fix 3: Comprehensive Sensor Validation (Issue #9)

Replace the `validate_sensor_data()` function with:

```python
REQUIRED_SENSOR_FIELDS = {
    "all": ["temperature", "humidity", "pH", "N", "P", "K", "rainfall", "soilMoisture"],
    "disease": ["temperature", "humidity", "pH", "N", "P", "K", "rainfall"],
    "irrigation": ["soilMoisture", "temperature", "humidity"],
    "yield": ["rainfall", "temperature"]
}

def validate_sensor_data(sensor, crop=None, model_type="all"):
    """Validate sensor data is complete and within acceptable ranges.
    
    Returns: (valid, errors_list)
    """
    errors = []
    
    if not isinstance(sensor, dict):
        errors.append("sensor must be a dict")
        return False, errors
    
    # Check all required fields are present
    required = REQUIRED_SENSOR_FIELDS.get(model_type, REQUIRED_SENSOR_FIELDS["all"])
    for key in required:
        if key not in sensor:
            errors.append(f"Missing required sensor field: {key}")
            continue
        
        val = sensor[key]
        if val is None:
            errors.append(f"{key} is null/None")
            continue
        
        if not isinstance(val, (int, float)):
            errors.append(f"{key} must be numeric (got {type(val).__name__})")
            continue
        
        # Check sensor ranges
        if key in SENSOR_RANGES:
            min_val, max_val = SENSOR_RANGES[key]
            
            # Use crop-specific range if available
            if crop and crop in CROP_SPECIFIC_RANGES and key in CROP_SPECIFIC_RANGES[crop]:
                min_val, max_val = CROP_SPECIFIC_RANGES[crop][key]
            
            if val < min_val or val > max_val:
                errors.append(f"{key}={val} out of range [{min_val}, {max_val}]")
    
    return len(errors) == 0, errors
```

---

## Fix 4: Add API Documentation Endpoint (Issue #10)

Add this endpoint to provide API documentation:

```python
@app.route("/api/docs", methods=["GET"])
def api_docs():
    """Return API documentation."""
    return jsonify({
        "service": "KISAAN AI - Precision Farming Intelligence",
        "version": "1.0.0",
        "endpoints": {
            "GET /crops": {
                "description": "Get list of supported crops for each model",
                "auth": "none",
                "response": {
                    "disease_model": ["list of crops"],
                    "irrigation_model": ["list of crops"],
                    "yield_model": ["list of crops"],
                    "cnn_scanner": ["list of crops"]
                }
            },
            "POST /predict": {
                "description": "Run all predictions on sensor data",
                "auth": "Firebase ID Token (Bearer token in Authorization header)",
                "request": {
                    "sensor": {
                        "temperature": "float (°C)",
                        "humidity": "float (0-100%)",
                        "soilMoisture": "float (0-1000)",
                        "rainfall": "float (mm)",
                        "pH": "float (0-14)",
                        "N": "float (mg/kg)",
                        "P": "float (mg/kg)",
                        "K": "float (mg/kg)"
                    },
                    "config": {
                        "cropType": "string (required)",
                        "country": "string (default: India)",
                        "cropDays": "int (1-365)",
                        "pesticides": "float",
                        "year": "int (1990-2100)"
                    }
                },
                "response": {
                    "disease": {
                        "label": "Healthy | At_Risk | Not Available",
                        "riskScore": "0-100",
                        "riskReasons": ["array of strings"]
                    },
                    "irrigation": {
                        "irrigate": "0 | 1",
                        "label": "Irrigate | No Irrigation",
                        "confidence": "0-1"
                    },
                    "yield": {
                        "kgPerHa": "float"
                    }
                }
            },
            "POST /predict/image": {
                "description": "Predict leaf disease from image",
                "auth": "Firebase ID Token",
                "request": {
                    "image": "base64 encoded image (JPEG/PNG, max 8MB, resized to 224x224)"
                },
                "response": {
                    "crop": "string",
                    "disease": "string",
                    "confidence": "float (0-100%)",
                    "is_healthy": "boolean",
                    "treatment": "string",
                    "top3": "array of alternatives"
                }
            }
        },
        "limits": {
            "/predict": "10 per minute per IP",
            "/predict/image": "5 per minute per IP",
            "max_payload_size": "12 MB",
            "max_image_size": "8 MB"
        }
    }), 200
```

---

## Fix 5: Add Authentication Audit Logging (Issue #11)

Add this global tracking for auth failures:

```python
# At module level
AUTH_FAILURE_TRACKING = {}  # {ip: [(timestamp, reason), ...]}
AUTH_LOCK = threading.Lock()

def track_auth_failure(ip, reason):
    """Track authentication failures for attack detection."""
    with AUTH_LOCK:
        if ip not in AUTH_FAILURE_TRACKING:
            AUTH_FAILURE_TRACKING[ip] = []
        
        AUTH_FAILURE_TRACKING[ip].append((datetime.utcnow(), reason))
        
        # Keep only last 1 hour of failures
        cutoff = datetime.utcnow() - timedelta(hours=1)
        AUTH_FAILURE_TRACKING[ip] = [
            (ts, r) for ts, r in AUTH_FAILURE_TRACKING[ip] if ts > cutoff
        ]
        
        # Alert if excessive failures
        if len(AUTH_FAILURE_TRACKING[ip]) >= 5:
            log.critical(
                "SECURITY ALERT: %d auth failures from IP %s in past hour. "
                "Possible brute force attack detected.",
                len(AUTH_FAILURE_TRACKING[ip]), ip
            )

# Update require_auth():
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if TEST_MODE:
            flask_env = os.getenv("FLASK_ENV", "development").lower()
            if flask_env == "production":
                log.critical(
                    "SECURITY ALERT: TEST_MODE=true in PRODUCTION! Rejecting request from %s",
                    request.remote_addr
                )
                return jsonify({"error": "Service temporarily unavailable"}), 503
            request.uid = "test_user_001"
            request.user_email = "test@kisaan.ai"
            return f(*args, **kwargs)
        
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            reason = "missing authorization header"
            track_auth_failure(request.remote_addr, reason)
            log.warning("Auth failure from %s: %s", request.remote_addr, reason)
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        
        id_token = auth_header[len("Bearer "):].strip()
        try:
            decoded = auth.verify_id_token(
                id_token,
                check_revoked=os.getenv("FIREBASE_CHECK_REVOKED", "true").lower() == "true"
            )
            request.uid = decoded["uid"]
            request.user_email = decoded.get("email", "unknown")
            
            # Clear failures on successful auth
            with AUTH_LOCK:
                if request.remote_addr in AUTH_FAILURE_TRACKING:
                    del AUTH_FAILURE_TRACKING[request.remote_addr]
            
        except Exception as e:
            reason = str(e)[:100]
            track_auth_failure(request.remote_addr, reason)
            log.warning("Token verification failed from %s: %s", request.remote_addr, reason)
            return jsonify({"error": "Invalid or expired token"}), 401
        
        return f(*args, **kwargs)
    return decorated
```

---

## Fix 6: HTTPS Enforcement in Frontend (Issue #12)

Update `frontend/firebase-config.example.js`:

```javascript
// ⚠️ SECURITY: ALWAYS USE HTTPS IN PRODUCTION
// HTTP EXPOSES AUTHENTICATION TOKENS AND SENSOR DATA IN CLEAR TEXT

// Detect environment and set appropriate backend URL
const isProduction = window.location.protocol === 'https:' || 
                     window.location.hostname === 'your-production-domain.com';

if (isProduction && window.location.protocol !== 'https:') {
    // Force redirect to HTTPS in production
    window.location.protocol = 'https:';
}

window.FIREBASE_CONFIG = {
    apiKey: "YOUR_API_KEY",
    authDomain: "YOUR_PROJECT.firebaseapp.com",
    databaseURL: "https://YOUR_PROJECT-default-rtdb.firebaseio.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID"
};

// Backend URL configuration
if (isProduction) {
    // Production MUST use HTTPS
    window.BACKEND_BASE_URL = "https://api.kisaan.com";
} else {
    // Development/local testing
    window.BACKEND_BASE_URL = "http://localhost:5000";
}

// Verify HTTPS in production
if (isProduction && !window.BACKEND_BASE_URL.startsWith('https://')) {
    console.error(
        "SECURITY ERROR: Backend URL must use HTTPS in production. " +
        "Current URL: " + window.BACKEND_BASE_URL
    );
    throw new Error("Insecure backend configuration detected");
}
```

---

## Fix 7: Clean Up Unused Models (Issue #13)

Remove or archive unused model files:

```bash
# Create archive directory
mkdir -p backend/models/archive

# Move unused models
mv backend/models/crop_disease_model.pkl backend/models/archive/
mv backend/models/disease_label_encoder.pkl backend/models/archive/
mv backend/models/crop_name_encoder.pkl backend/models/archive/
mv backend/models/crop_recommendation_model.pkl backend/models/archive/
mv backend/models/disease_name_label_encoder.pkl backend/models/archive/

# Update .gitignore
echo "backend/models/archive/" >> .gitignore
```

Remove these lines from `backend/app.py`:

```python
# DELETE THESE LINES (they're not used anymore):
disease_model     = joblib.load(os.path.join(BASE, "models", "crop_disease_model.pkl"))
disease_le        = joblib.load(os.path.join(BASE, "models", "disease_label_encoder.pkl"))
crop_name_le      = joblib.load(os.path.join(BASE, "models", "crop_name_encoder.pkl"))
```

---

## Deployment Checklist with Fixes

```bash
# 1. Update dependencies
pip install -r backend/requirements.txt

# 2. Set environment variables for production
export FLASK_ENV=production
export CORS_ALLOWED_ORIGINS="https://myapp.com,https://api.myapp.com"
export FIREBASE_KEY_PATH="/path/to/firebase-key.json"
export FIREBASE_DB_URL="https://myproject.firebaseio.com"
export FIREBASE_CHECK_REVOKED=true

# 3. Test rate limiting (if using Redis)
redis-server &  # Make sure Redis is running

# 4. Run security checks
pip install bandit
bandit -r backend/ -ll

# 5. Start with Gunicorn (production WSGI server)
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app

# 6. Configure SSL/TLS with nginx reverse proxy
# (See deployment guide for nginx config)
```

---

## Testing Fixes

```python
# Test rate limiting
import time
import requests

headers = {"Authorization": "Bearer test_token"}
for i in range(15):
    response = requests.get("http://localhost:5000/status", headers=headers)
    if i == 10:
        print(f"Request 11: {response.status_code} - Should be 429 (Too Many Requests)")
    else:
        print(f"Request {i+1}: {response.status_code}")

# Test auth failure tracking
for i in range(7):
    requests.get("http://localhost:5000/status")  # No auth header
    time.sleep(0.1)
    
# Check logs for "SECURITY ALERT: 5 auth failures..."

# Test CORS validation
os.environ["FLASK_ENV"] = "production"
os.environ["CORS_ALLOWED_ORIGINS"] = ""  # Should raise error on startup
```

---

## Summary

These fixes address:
- ✅ Rate limiting to prevent DOS attacks
- ✅ Log injection prevention  
- ✅ Complete sensor data validation
- ✅ API documentation for developers
- ✅ Authentication audit logging
- ✅ HTTPS enforcement
- ✅ Unused model cleanup

**Next Steps:**
1. Apply fixes in order of priority (rate limiting → auth logging → validation)
2. Test each fix with the provided test code
3. Update production deployment documentation
4. Run security audit tools (bandit, semgrep)
5. Conduct penetration testing before production release


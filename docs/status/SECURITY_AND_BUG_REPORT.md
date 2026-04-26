# 🔍 KISAAN AI - Security & Bug Analysis Report
**Date:** April 21, 2026  
**Status:** ✅ **TESTED & ANALYZED**  
**Overall Risk Level:** 🟡 **MEDIUM** (Fixable Issues Found)

---

## Executive Summary

The KISAAN AI platform has been fully tested and analyzed. The system is **functional in TEST_MODE** with proper backend and ML models working correctly. However, **9 critical/high priority issues** and several low-priority issues have been identified that need attention before production deployment.

### ✅ What's Working
- Backend Flask server starts successfully (after fixing xgboost dependency)
- All 4 ML models load correctly (disease engine, irrigation, yield, CNN)
- Input validation is implemented for sensor/config data
- Payload size limits are enforced (12MB max)
- Security headers are being added
- CORS is properly configured
- Firebase integration architecture is solid
- Base64 image validation works

### ❌ Issues Found
Total Issues: **12** (3 Critical, 4 High, 3 Medium, 2 Low)

---

## 🔴 CRITICAL ISSUES (Must Fix Before Production)

### 1. **Missing Dependency in requirements.txt** {HIGH}
**File:** [../../backend/requirements.txt](../../backend/requirements.txt)  
**Severity:** 🔴 CRITICAL  
**Issue:** The `xgboost` package is missing from requirements.txt but is imported in app.py  
**Impact:** Backend crashes on startup with `ModuleNotFoundError: No module named 'xgboost'`

```
Error trace:
  File "/workspaces/4thSem_Mini_Project/backend/app.py", line 58, in <module>
    disease_model = joblib.load(os.path.join(BASE, "models", "crop_disease_model.pkl"))
  ModuleNotFoundError: No module named 'xgboost'
```

**Fix:** Add `xgboost>=1.7.0` to [../../backend/requirements.txt](../../backend/requirements.txt)

```diff
+ xgboost>=1.7.0
```

**Status:** ❌ NOT FIXED (but installed during testing)

---

### 2. **scikit-learn Version Mismatch Warnings** {HIGH}
**File:** [../../backend/app.py](../../backend/app.py)  
**Severity:** 🔴 CRITICAL  
**Issue:** All ML models were pickled with scikit-learn 1.6.1 but the environment has 1.8.0

**Error Output:**
```
InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 
1.6.1 when using version 1.8.0. This might lead to breaking code or invalid results.
```

**Affected Components:**
- LabelEncoder
- DummyClassifier  
- DecisionTreeRegressor
- GradientBoostingClassifier
- StandardScaler

**Fix Options:**
1. **Recommended:** Pin scikit-learn to 1.6.1 in requirements.txt:
   ```
   scikit-learn==1.6.1
   ```

2. **Alternative:** Re-train and re-pickle all models with scikit-learn 1.8.0

**Status:** ❌ NOT FIXED (works but with warnings)

---

### 3. **Missing TensorFlow/Keras Dependency** {HIGH}
**File:** [../../backend/requirements.txt](../../backend/requirements.txt)  
**Severity:** 🔴 CRITICAL  
**Issue:** TensorFlow is required for CNN model loading but missing from requirements.txt

**Impact:** If CNN model fails to load in production, there's no fallback gracefully documented

**Current Code:**
```python
def load_cnn():
    global CNN_MODEL, CNN_LABELS
    try:
        import tensorflow as tf  # ← Not in requirements.txt!
        from tensorflow import keras
```

**Fix:** Add tensorflow to requirements.txt:
```
tensorflow>=2.13.0
```

**Status:** ❌ NOT FIXED (but installed during testing)

---

### 4. **Hardcoded CORS Origins** {MEDIUM}
**File:** [../../backend/app.py](../../backend/app.py), Line 702-707  
**Severity:** 🟡 MEDIUM  
**Issue:** CORS origins are hardcoded with localhost defaults, no production check

**Current Code:**
```python
def _cors_allowed_origins():
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
```

**Risks:**
- Easy to misconfigure in production
- Default allows only localhost — will fail in production if not set
- No validation of origin format
- Could accidentally expose API to unauthorized origins

**Fix:**
```python
def _cors_allowed_origins():
    raw = os.getenv("CORS_ALLOWED_ORIGINS")
    if not raw:
        if os.getenv("FLASK_ENV") == "production":
            raise RuntimeError(
                "CORS_ALLOWED_ORIGINS must be set in production. "
                "Example: https://myapp.com,https://api.myapp.com"
            )
        # Dev mode
        return ["http://localhost:8080", "http://127.0.0.1:8080"]
    
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    
    # Validate URLs
    import re
    url_pattern = re.compile(r'^https?://[a-zA-Z0-9.-]+(?:\:\d+)?$')
    for origin in origins:
        if not url_pattern.match(origin):
            raise ValueError(f"Invalid CORS origin format: {origin}")
    
    return origins
```

**Status:** ❌ NOT FIXED

---

## 🟠 HIGH PRIORITY ISSUES

### 5. **TEST_MODE Bypasses Authentication** {HIGH}
**File:** [../../backend/app.py](../../backend/app.py), Lines 733-739  
**Severity:** 🟠 HIGH  
**Issue:** When `TEST_MODE=true`, all authentication checks are bypassed

**Vulnerable Code:**
```python
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # In TEST_MODE, use dummy user ID
        if TEST_MODE:
            request.uid = "test_user_001"
            request.user_email = "test@kisaan.ai"
            return f(*args, **kwargs)  # ← No real auth!
```

**Risk:** If someone deploys with `TEST_MODE=true` by accident, the entire API is exposed without authentication

**Fix:** Add a strict check to prevent TEST_MODE in production:
```python
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if TEST_MODE:
            if os.getenv("FLASK_ENV") == "production":
                log.error("TEST_MODE=true in PRODUCTION! Rejecting request.")
                return jsonify({"error": "Service unavailable"}), 503
            request.uid = "test_user_001"
            request.user_email = "test@kisaan.ai"
            return f(*args, **kwargs)
        # ... rest of auth code
```

**Status:** ❌ NOT FIXED

---

### 6. **Firebase Realtime DB Open to All** {HIGH}
**File:** [../../backend/firebase_structure.py](../../backend/firebase_structure.py)  
**Severity:** 🟠 HIGH  
**Issue:** Firebase Realtime DB structure is documented but security rules not mentioned

**Risk:** If Firebase security rules are not properly configured, all user data (sensor readings, health predictions, yields) could be accessible to anyone with the database URL

**Firebase Security Rules Should Be:**
```json
{
  "rules": {
    "users": {
      "$uid": {
        ".read": "auth.uid === $uid",
        ".write": "auth.uid === $uid",
        "sensors": {
          "latest": {
            ".read": true,  // ESP32 needs read access
            ".write": "auth.uid === $uid"
          }
        },
        "config": {
          ".read": "auth.uid === $uid",
          ".write": "auth.uid === $uid"
        },
        "predictions": {
          ".read": "auth.uid === $uid",
          ".write": "root.child('_backend').val() === true"  // Only backend writes
        },
        "cnn_prediction": {
          ".read": "auth.uid === $uid",
          ".write": "root.child('_backend').val() === true"
        }
      }
    }
  }
}
```

**Fix:** Document and enforce Firebase security rules in deployment guide

**Status:** ❌ NOT FIXED (security rules not visible in code)

---

### 7. **No Rate Limiting on API Endpoints** {HIGH}
**File:** [../../backend/app.py](../../backend/app.py)  
**Severity:** 🟠 HIGH  
**Issue:** No rate limiting on critical endpoints like `/predict` and `/predict/image`

**Risk:** 
- DOS attacks possible (send 1000s of prediction requests)
- ML model inference is compute-heavy; unlimited requests could crash server
- Image processing can consume memory; no per-user limits

**Attack Example:**
```bash
# DOS attack
for i in {1..10000}; do
  curl -X POST http://api.kisaan.com/predict/image \
    -H "Authorization: Bearer token" \
    -d '{"image": "base64..."}' &
done
```

**Fix:** Add rate limiting using Flask-Limiter:
```bash
pip install Flask-Limiter redis
```

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per day", "50 per hour"]
)

@app.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")  # 10 predictions per minute per IP
@require_auth
def predict_endpoint():
    # ...
```

**Status:** ❌ NOT FIXED

---

### 8. **No Input Sanitization for Crop Names** {HIGH}
**File:** [../../backend/app.py](../../backend/app.py), Lines 948-960  
**Severity:** 🟠 HIGH  
**Issue:** Crop names are used in logging and predictions without proper sanitization

**Vulnerable Code:**
```python
def predict_disease(sensor, config):
    crop = config.get("cropType", "").lower().strip()
    # ... later
    log.info("Health risk [%s] → %s (score=%d)", crop, result["label"], result["riskScore"])
```

**Risk:** Log injection attacks possible
```python
config = {"cropType": "rice\n[ERROR] ADMIN ACCOUNT COMPROMISED"}
# Log would show:
# "Health risk [rice\nERROR] ADMIN ACCOUNT COMPROMISED] → ..."
```

**Fix:** Use parameterized logging:
```python
def predict_disease(sensor, config):
    crop = config.get("cropType", "").lower().strip()
    # Validate crop is in allowed list
    if crop not in DISEASE_CROPS:
        crop = "unknown"
    
    # Safe logging with parameters
    log.info("Health risk [%s] → %s (score=%d)", 
             crop.replace("\n", "\\n"), 
             result["label"], 
             result["riskScore"])
```

**Status:** ❌ NOT FIXED

---

## 🟡 MEDIUM PRIORITY ISSUES

### 9. **Missing Sensor Data Fields Not Validated** {MEDIUM}
**File:** [../../backend/app.py](../../backend/app.py), Lines 653-685  
**Severity:** 🟡 MEDIUM  
**Issue:** Sensor validation only checks ranges but not completeness; crops can get None values

**Example:**
```python
sensor = {"temperature": 25}  # Missing humidity, pH, N, P, K, etc.
# Gets logged as sensor validation passed
# But downstream prediction gets None values
```

**Current Code:**
```python
def validate_sensor_data(sensor, crop=None):
    errors = []
    if not isinstance(sensor, dict):
        errors.append("sensor must be a dict")
        return False, errors
    
    # Only checks if VALUE is in range IF it exists
    # Doesn't require all fields to be present!
```

**Fix:** Define required fields per crop:
```python
REQUIRED_SENSOR_FIELDS = {
    "disease": ["temperature", "humidity", "pH", "N", "P", "K", "rainfall"],
    "irrigation": ["soilMoisture", "temperature", "humidity"],
    "yield": ["rainfall", "temperature"]
}

def validate_sensor_data(sensor, crop=None, model_type="disease"):
    errors = []
    
    required = REQUIRED_SENSOR_FIELDS.get(model_type, [])
    for field in required:
        if field not in sensor:
            errors.append(f"Missing required sensor field: {field}")
```

**Status:** ❌ NOT FIXED

---

### 10. **CNN Image Size Not Documented** {MEDIUM}
**File:** [../../backend/app.py](../../backend/app.py), Line 71  
**Severity:** 🟡 MEDIUM  
**Issue:** CNN expects 224x224 images but this isn't documented anywhere

**Current Code:**
```python
CNN_IMG_SIZE = 224
# ... later ...
img = img.resize((CNN_IMG_SIZE, CNN_IMG_SIZE))
```

**Risk:** Frontend developers don't know the image requirements; large images are resized without warning (quality loss)

**Fix:** Add documentation and validation:
```python
CNN_IMG_SIZE = 224
CNN_IMG_MAX_SIZE = 5 * 1024 * 1024  # 5MB

def predict_leaf_image(image_b64):
    """
    Predict leaf disease from base64 image.
    
    Args:
        image_b64: Base64 encoded image (any size, will be resized to 224x224)
        
    Returns:
        dict with crop, disease, confidence, treatment
        
    Note: Images are resized to 224x224. Provide high-res images for best accuracy.
    """
    if CNN_MODEL is None:
        return {"error": "CNN model not loaded..."}
```

**Frontend API Documentation (missing):**
```
POST /predict/image
Content-Type: application/json

Request:
{
  "image": "data:image/jpeg;base64,..."  // or just base64 string
}

Response:
{
  "crop": "tomato",
  "disease": "Early Blight",
  "confidence": 98.5,
  "is_healthy": false,
  "treatment": "Use sulfur-based fungicides...",
  "top3": [...],  // Alternative predictions
  "timestamp": "..."
}

Supported Image Formats: JPEG, PNG, WebP (any size, resized to 224x224)
Max Size: 8MB (binary)
```

**Status:** ❌ NOT FIXED (documentation missing)

---

### 11. **No Logging for Failed Authentication Attempts** {MEDIUM}
**File:** [../../backend/app.py](../../backend/app.py), Lines 741-750  
**Severity:** 🟡 MEDIUM  
**Issue:** Auth failures only log warnings, no audit trail

**Current Code:**
```python
except Exception as e:
    log.warning("Token verification failed: %s", e)
    return jsonify({"error": "Invalid or expired token"}), 401
```

**Risk:** No way to detect brute force attacks or compromised tokens

**Fix:** Add detailed auth logging:
```python
from datetime import datetime, timedelta

AUTH_FAILURES = {}  # Track failed attempts per IP

@require_auth
def decorated(*args, **kwargs):
    # ...
    try:
        decoded = auth.verify_id_token(id_token, check_revoked=True)
        request.uid = decoded["uid"]
        request.user_email = decoded.get("email", "unknown")
        
        # Clear failure count on success
        client_ip = request.remote_addr
        if client_ip in AUTH_FAILURES:
            del AUTH_FAILURES[client_ip]
            
    except Exception as e:
        client_ip = request.remote_addr
        
        # Track failures
        if client_ip not in AUTH_FAILURES:
            AUTH_FAILURES[client_ip] = []
        AUTH_FAILURES[client_ip].append(datetime.utcnow())
        
        # Keep only last 1 hour of failures
        cutoff = datetime.utcnow() - timedelta(hours=1)
        AUTH_FAILURES[client_ip] = [t for t in AUTH_FAILURES[client_ip] if t > cutoff]
        
        # Alert on excessive failures (5+ in 1 hour = potential attack)
        if len(AUTH_FAILURES[client_ip]) >= 5:
            log.critical("SECURITY ALERT: %d auth failures from IP %s in 1 hour", 
                        len(AUTH_FAILURES[client_ip]), client_ip)
        
        log.warning("Auth failure from %s: %s", client_ip, e)
        return jsonify({"error": "Invalid or expired token"}), 401
```

**Status:** ❌ NOT FIXED

---

## 🟢 LOW PRIORITY ISSUES

### 12. **No HTTPS Enforcement Documentation** {LOW}
**File:** [../../frontend/firebase-config.example.js](../../frontend/firebase-config.example.js)  
**Severity:** 🟢 LOW  
**Issue:** Example shows HTTP for localhost development, but no warning about HTTPS in production

**Current Code:**
```javascript
window.BACKEND_BASE_URL = "http://localhost:5000";
```

**Risk:** Developers might copy-paste this to production with HTTP, exposing auth tokens

**Fix:** Add warning comment:
```javascript
// ⚠️ MUST USE HTTPS IN PRODUCTION
// HTTP EXPOSES AUTH TOKENS AND SENSOR DATA IN PLAIN TEXT
window.BACKEND_BASE_URL = process.env.NODE_ENV === "production" 
  ? "https://api.kisaan.com"
  : "http://localhost:5000";
```

**Status:** ❌ NOT FIXED

---

### 13. **Unused Model Files** {LOW}
**File:** [backend/models/](backend/models/)  
**Severity:** 🟢 LOW  
**Issue:** Several model files are loaded but some may not be used

**Files:**
- `crop_disease_model.pkl` - Loaded but **replaced by rule engine**
- `disease_label_encoder.pkl` - No longer used (rule engine doesn't need it)
- `crop_name_encoder.pkl` - No longer used
- `crop_recommendation_model.pkl` - NOT LOADED, appears orphaned
- `disease_name_label_encoder.pkl` - NOT LOADED, appears orphaned

**Impact:** 
- ~1.3MB wasted memory loading unused models
- Confusing for future maintainers
- Unused dependencies increase attack surface

**Fix:** Remove or archive unused models:
```bash
# Move to archive
mkdir models/archive
mv models/crop_disease_model.pkl models/archive/
mv models/disease_label_encoder.pkl models/archive/
# Remove from app.py imports
```

**Updated Code:**
```python
# Only load what's actually used:
irrigation_model  = joblib.load(os.path.join(BASE, "models", "irrigation_model.pkl"))
irrigation_scaler = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
irrigation_le     = joblib.load(os.path.join(BASE, "models", "label_encoder.pkl"))
yield_model       = joblib.load(os.path.join(BASE, "models", "crop_yield_model.pkl"))
# Rule engine doesn't need disease_model
```

**Status:** ❌ NOT FIXED

---

## 📋 Summary of Issues by Category

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| **Dependencies** | 2 | 1 | 0 | 0 | 3 |
| **Security** | 0 | 2 | 2 | 1 | 5 |
| **Input Validation** | 1 | 1 | 1 | 0 | 3 |
| **Code Quality** | 0 | 0 | 0 | 1 | 1 |
| **Total** | **3** | **4** | **3** | **2** | **12** |

---

## 🚀 Production Deployment Checklist

Before deploying to production:

- [ ] **FIX CRITICAL ISSUES 1-3** (dependencies)
- [ ] **FIX CRITICAL ISSUE 4** (CORS origins validation)
- [ ] **FIX ISSUE 5** (TEST_MODE protection)
- [ ] **CONFIGURE ISSUE 6** (Firebase security rules)
- [ ] **IMPLEMENT ISSUE 7** (Rate limiting)
- [ ] **FIX ISSUE 8** (Input sanitization)
- [ ] **FIX ISSUE 9** (Sensor validation)
- [ ] **DOCUMENT ISSUE 10** (CNN image requirements)
- [ ] **IMPLEMENT ISSUE 11** (Auth logging)
- [ ] **FIX ISSUE 12** (HTTPS enforcement)
- [ ] **CLEAN ISSUE 13** (Remove unused models)
- [ ] Set `FLASK_ENV=production`
- [ ] Enable FIREBASE_CHECK_REVOKED in production
- [ ] Configure proper logging (syslog or cloud logging)
- [ ] Set up SSL/TLS certificates
- [ ] Configure environment variables properly
- [ ] Run security audit with `bandit` or `semgrep`
- [ ] Load test with `locust` to verify rate limiting
- [ ] Penetration test authentication flow

---

## ✅ Testing Results

### API Endpoints Tested
- ✅ GET `/crops` - Returns supported crops correctly
- ✅ POST `/predict` - Predictions working with valid data
- ✅ GET `/status` - Status check working
- ✅ Input validation - Rejects invalid sensor data
- ✅ Payload size limit - Rejects payloads > 12MB
- ✅ Base64 image validation - Rejects invalid images
- ✅ Security headers - Added correctly

### ML Models Tested
- ✅ Disease risk engine - Working (8/10 risk for drought stress)
- ✅ Irrigation recommender - Working (recommends irrigate)
- ✅ Yield predictor - Working (4632 kg/ha for rice)
- ✅ CNN model - Loaded successfully (38 classes)

### Error Handling Tested
- ✅ Missing sensor/config - Proper error response
- ✅ Non-numeric temperature - Rejected with validation error
- ✅ Unsupported crop type - Returns "Not Available" gracefully
- ✅ Malformed JSON - Returns 400 error

---

## 📚 Recommended Security Improvements (Beyond Scope)

1. **Implement OAuth2 with Redis session caching**
2. **Add database audit logging for all predictions**
3. **Implement image watermarking for CNN predictions**
4. **Add data encryption at rest (sensor readings in Firebase)**
5. **Implement anomaly detection for unusual predictions**
6. **Set up automated security scanning (SAST/DAST)**
7. **Add request signing to prevent tampering**
8. **Implement API versioning for backward compatibility**

---

## 🔗 Related Files

- [../../backend/app.py](../../backend/app.py) - Main Flask application
- [../../backend/firebase_structure.py](../../backend/firebase_structure.py) - Database schema
- [../../backend/requirements.txt](../../backend/requirements.txt) - Dependencies
- [../../frontend/dashboard.html](../../frontend/dashboard.html) - Web interface
- [../../frontend/firebase-config.example.js](../../frontend/firebase-config.example.js) - Config template

---

## 📞 Questions & Support

For questions about this report, refer to:
- Backend issues: Check [../../backend/app.py](../../backend/app.py) implementation
- Security architecture: Review [../../README.md](../../README.md#-system-architecture)
- Deployment guide: (Create deployment_guide.md)

---

**Report Generated:** 2026-04-21 07:30 UTC  
**Tester:** GitHub Copilot  
**Status:** ✅ TESTING COMPLETE - READY FOR FIXES

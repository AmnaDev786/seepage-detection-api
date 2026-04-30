from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import requests
import os
import io
import tempfile
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==========================================
# GITHUB MODEL URLs 
# (Replace YOUR_USERNAME and YOUR_REPO with your actual GitHub details)
# ==========================================
# ==========================================
# GITHUB MODEL URLs — Updated for AmnaDev786
# ==========================================
# Note: Using 'main' as the branch name. 
# If your default branch is 'master', change 'main' to 'master'.
GITHUB_BASE = "https://raw.githubusercontent.com/AmnaDev786/seepage-detection-api/main/models/"

MODEL_URLS = {
    "det_model":  GITHUB_BASE + "final_detection_model.h5",
    "det_scaler": GITHUB_BASE + "detection_csi_scaler.pkl",
    "loc_model":  GITHUB_BASE + "final_localization_model.h5",
    "loc_scaler": GITHUB_BASE + "csi_loc_scaler.pkl",
}

# ==========================================
# CONFIGURATION
# ==========================================
WINDOW_SIZE     = 50
STEP_SIZE        = 5
PROB_THRESHOLD  = 0.98
MIN_WET_WINDOWS = 40
MIN_INTENSITY   = 5

X_POINTS = [-2, -1, 0, 1, 2]
Y_POINTS  = [1, 0, -1]

# Cache models in memory to avoid re-downloading per request
_models = {}

def download_file(url, suffix):
    """Download a file from GitHub and save to a temporary location."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(r.content)
    tmp.close()
    return tmp.name

def load_models():
    """Download and load models if they aren't already in memory."""
    global _models
    if _models:
        return _models

    print("Fetching models from GitHub...")

    det_path = download_file(MODEL_URLS["det_model"],  ".h5")
    loc_path = download_file(MODEL_URLS["loc_model"],  ".h5")
    ds_path  = download_file(MODEL_URLS["det_scaler"], ".pkl")
    ls_path  = download_file(MODEL_URLS["loc_scaler"], ".pkl")

    _models["det_model"]  = tf.keras.models.load_model(det_path, compile=False)
    _models["loc_model"]  = tf.keras.models.load_model(loc_path, compile=False)
    with open(ds_path, "rb") as f: _models["det_scaler"] = pickle.load(f)
    with open(ls_path, "rb") as f: _models["loc_scaler"] = pickle.load(f)

    # Cleanup temp storage
    for p in [det_path, loc_path, ds_path, ls_path]:
        os.unlink(p)

    print("Models synchronized successfully.")
    return _models

def parse_csi(csi_str):
    try:
        csi_str = str(csi_str).replace('"', '').strip()
        vals = [int(x) for x in csi_str.split(',') if x]
        return vals if len(vals) == 384 else None
    except:
        return None

# ==========================================
# FIREBASE INTEGRATION (Firestore REST API)
# ==========================================
# Replace YOUR_PROJECT_ID with your Firebase Project ID
FIREBASE_URL = "https://firestore.googleapis.com/v1/projects/YOUR_PROJECT_ID/databases/(default)/documents/scans"

def save_to_firebase(result_data):
    """Log the scan to Firestore for the History screen."""
    try:
        doc = {
            "fields": {
                "timestamp":  {"stringValue": result_data["timestamp"]},
                "status":      {"stringValue": result_data["status"]},
                "best_x":      {"doubleValue": result_data.get("best_x", 0)},
                "best_y":      {"doubleValue": result_data.get("best_y", 0)},
                "wet_count":   {"integerValue": str(result_data.get("wet_count", 0))},
                "moisture":    {"doubleValue": result_data.get("moisture_index", 0)},
            }
        }
        requests.post(FIREBASE_URL, json=doc, timeout=5)
    except Exception as e:
        print(f"Sync error: {e}")

# ==========================================
# UNITY ENDPOINTS
# ==========================================

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    file = request.files["file"]
    try:
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))

        if "csi_data" not in df.columns:
            return jsonify({"error": "Invalid CSV format"}), 400

        df["csi_parsed"] = df["csi_data"].apply(parse_csi)
        csi_matrix = np.array(df.dropna(subset=["csi_parsed"])["csi_parsed"].tolist())

        if len(csi_matrix) < WINDOW_SIZE:
            return jsonify({"error": "Insufficient data points"}), 400

        models = load_models()
        probs  = []
        coords = []

        # Sliding Window Analysis
        for i in range(0, len(csi_matrix) - WINDOW_SIZE + 1, STEP_SIZE):
            window = csi_matrix[i: i + WINDOW_SIZE]

            # Stage 1: Detection
            in_det = models["det_scaler"].transform(window.reshape(-1, 384)).reshape(1, WINDOW_SIZE, 384)
            p = float(models["det_model"].predict(in_det, verbose=0)[0][0])
            probs.append(p)

            # Stage 2: Localization
            if p > PROB_THRESHOLD:
                in_loc = models["loc_scaler"].transform(window.reshape(-1, 384)).reshape(1, WINDOW_SIZE, 384)
                xy = models["loc_model"].predict(in_loc, verbose=0)[0].tolist()
                coords.append(xy)

        # Heatmap Calculation
        h_map = [[0.0] * len(X_POINTS) for _ in range(len(Y_POINTS))]
        for xp, yp in coords:
            xi = int(np.argmin([abs(xp - v) for v in X_POINTS]))
            yi = int(np.argmin([abs(yp - v) for v in Y_POINTS]))
            h_map[yi][xi] += 1

        # Apply Intensity Filter
        for yi in range(len(Y_POINTS)):
            for xi in range(len(X_POINTS)):
                if h_map[yi][xi] < MIN_INTENSITY:
                    h_map[yi][xi] = 0.0

        wet_count = len(coords)
        is_wet    = wet_count >= MIN_WET_WINDOWS

        result = {
            "status":    "WET" if is_wet else "DRY",
            "wet_count": wet_count,
            "timestamp": datetime.now().strftime("%b %d, %Y  %H:%M"),
            "heatmap":   h_map,
            "best_x":    0.0,
            "best_y":    0.0,
            "signal_strength": round(float(np.mean(probs)) * -100, 1),
            "moisture_index":  round((wet_count / max(len(probs), 1)) * 100, 1),
        }

        if is_wet and coords:
            flat   = np.array(h_map)
            max_ij = np.unravel_index(flat.argmax(), flat.shape)
            result["best_x"] = float(X_POINTS[max_ij[1]])
            result["best_y"] = float(Y_POINTS[max_ij[0]])

        # Log Result to Cloud
        save_to_firebase(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def history():
    """Fetch previous scans for the Dashboard Screen."""
    try:
        r = requests.get(FIREBASE_URL + "?pageSize=10", timeout=5)
        data = r.json()
        docs = data.get("documents", [])
        results = []
        for doc in docs:
            f = doc.get("fields", {})
            results.append({
                "timestamp": f.get("timestamp", {}).get("stringValue", ""),
                "status":    f.get("status",    {}).get("stringValue", ""),
                "best_x":    f.get("best_x",    {}).get("doubleValue", 0),
                "best_y":    f.get("best_y",    {}).get("doubleValue", 0),
            })
        return jsonify({"history": results})
    except Exception as e:
        return jsonify({"history": [], "error": str(e)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
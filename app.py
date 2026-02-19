from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# --- Dynamic Model Loading ---
class FactoryModel:
    def __init__(self):
        try:
            # Try to load real model
            artifacts = joblib.load("FactoryGuard.joblib")
            self.model = artifacts['model']
            self.features = artifacts['features']
            self.is_real = True
            print("✅ Real FactoryGuard Model Loaded.")
        except:
            # Fallback Logic: Creates real probability changes based on data
            self.features = ["vibration", "temperature", "pressure"]
            self.is_real = False
            print("⚠️ Model file not found. Using Dynamic Logic Fallback.")

    def predict(self, input_dict):
        # Extract values in order, default to 0 if missing
        vals = [float(input_dict.get(f, 0)) for f in self.features]
        
        if self.is_real:
            input_data = np.array([vals])
            prob = self.model.predict_proba(input_data)[0, 1]
        else:
            # DYNAMIC CALCULATION: (Simulating a real model)
            # High values = High probability
            v_score = vals[0] / 10   # Assume 10 is max vibration
            t_score = vals[1] / 120  # Assume 120 is max temp
            p_score = vals[2] / 200  # Assume 200 is max pressure
            prob = min(0.99, (v_score + t_score + p_score) / 3)
            
        return round(float(prob) * 100, 2)

# Initialize the model wrapper
guard = FactoryModel()

@app.route('/')
def index():
    return render_template('index.html', features=guard.features)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        data = json.load(file)
        
        prob = guard.predict(data)
        
        return jsonify({
            "filename": file.filename,
            "probability": prob,
            "status": "CRITICAL" if prob > 70 else "STABLE"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

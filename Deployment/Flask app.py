from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and feature list
try:
    artifacts = joblib.load("factory_guard_model.pkl")
    model = artifacts['model']
    features = artifacts['features']
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback for testing if file doesn't exist yet
    features = ["vibration", "temperature", "pressure"]

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Ensure data is in the correct order for the model
        input_data = np.array([[data.get(f, 0) for f in features]])
        
        # Get probability from XGBoost
        prob = model.predict_proba(input_data)[0, 1]
        
        return jsonify({
            "probability": round(float(prob) * 100, 2),
            "status": "CRITICAL" if prob > 0.7 else "STABLE"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
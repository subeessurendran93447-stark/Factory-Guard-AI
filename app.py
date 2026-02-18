from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier # Added for fallback

app = Flask(__name__)

# Load the saved model and feature list
try:
    artifacts = joblib.load("FactoryGuard.joblib")
    model = artifacts['model']
    features = artifacts['features']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback configuration
    features = ["vibration", "temperature", "pressure"]
    # Create a dummy model so 'model' is always defined
    model = DummyClassifier(strategy="uniform")
    # Fit with dummy data (3 features)
    model.fit(np.zeros((10, 3)), np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Ensure data is in the correct order and converted to float
        input_data = np.array([[float(data.get(f, 0)) for f in features]])
        
        # Get probability
        prob = model.predict_proba(input_data)[0, 1]
        
        return jsonify({
            "probability": round(float(prob) * 100, 2),
            "status": "CRITICAL" if prob > 0.7 else "STABLE"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
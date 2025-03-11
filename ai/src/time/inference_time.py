import torch
import torch.nn.functional as F
import numpy as np
import pickle
from flask import Flask, request, jsonify

from train_time import StudyDurationPredictor, map_class_to_duration

app = Flask(__name__)

# 1) Load Model and Scaler
model = StudyDurationPredictor(input_dim=3)
model.load_state_dict(torch.load("study_duration_model.pth", map_location=torch.device('cpu')))
model.eval()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 2) Helper: Convert numeric minutes to model intensity
def map_minutes_to_intensity(minutes):
    """
    <30 -> class 0 (intensity 0.0)
    30-60 -> class 1 (intensity 1/3)
    60-120 -> class 2 (intensity 2/3)
    >=120 -> class 3 (intensity 1.0)
    """
    if minutes < 30:
        cls = 0
    elif minutes < 60:
        cls = 1
    elif minutes < 120:
        cls = 2
    else:
        cls = 3
    return cls / 3.0

# 3) Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON with keys:
      - "study_time" (float, minutes)
      - "break_time" (float, minutes)
      - "schedule_satisfaction" (float)
    Returns JSON with predicted class, duration, and probabilities.
    """
    try:
        data = request.get_json()
        study_time = float(data['study_time'])
        break_time = float(data['break_time'])
        schedule_satisfaction = float(data['schedule_satisfaction'])
    except (KeyError, ValueError):
        return jsonify({
            "error": "Invalid or missing keys. Expect study_time, break_time, schedule_satisfaction."
        }), 400

    # Convert numeric study_time to normalized intensity
    study_intensity = map_minutes_to_intensity(study_time)

    # Construct features [study_intensity, break_time, schedule_satisfaction]
    features = np.array([[study_intensity, break_time, schedule_satisfaction]], dtype=np.float32)
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    # Get model prediction
    with torch.no_grad():
        logits = model(features_tensor)
        probabilities = F.softmax(logits, dim=1).numpy()[0]
        predicted_class = int(np.argmax(probabilities))

    # Map predicted class to a duration label
    predicted_duration = map_class_to_duration(predicted_class)

    return jsonify({
        "predicted_class": predicted_class,
        "predicted_duration": predicted_duration,
        "probabilities": probabilities.tolist()
    })

if __name__ == '__main__':
    # By default, Flask runs on port 5000
    app.run(debug=True)

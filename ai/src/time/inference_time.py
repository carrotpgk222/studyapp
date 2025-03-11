import torch
import torch.nn.functional as F
import numpy as np
import pickle
from flask import Flask, request, jsonify

# Import your model and mapping function from train_time
from train_time import StudyDurationPredictor, map_class_to_duration

app = Flask(__name__)

# 1) Load Model and Scaler
model = StudyDurationPredictor(input_dim=3)
model.load_state_dict(torch.load("study_duration_model.pth", map_location=torch.device('cpu')))
model.eval()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 2) Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON with keys:
      - "review_id" (optional, integer)
      - "sessionDuration" (float, in minutes)
      - "breakTime" (float, in minutes)
      - "scheduleSatisfaction" (float)
    Converts minutes to hours and returns JSON with predicted class, duration, and probabilities.
    """
    try:
        data = request.get_json()
        review_id = data.get('review_id', None)
        session_duration = float(data['sessionDuration'])
        break_time = float(data['breakTime'])
        schedule_satisfaction = float(data['scheduleSatisfaction'])
    except (KeyError, ValueError):
        return jsonify({
            "error": "Invalid or missing keys. Expect sessionDuration, breakTime, scheduleSatisfaction."
        }), 400

    # Convert minutes to hours
    session_duration_hours = session_duration / 60.0
    break_time_hours = break_time / 60.0

    # Construct features [session_duration_hours, break_time_hours, schedule_satisfaction]
    features = np.array([[session_duration_hours, break_time_hours, schedule_satisfaction]], dtype=np.float32)
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    # Get model prediction
    with torch.no_grad():
        logits = model(features_tensor)
        probabilities = F.softmax(logits, dim=1).numpy()[0]
        predicted_class = int(np.argmax(probabilities))

    predicted_duration = map_class_to_duration(predicted_class)

    response = {
        "predicted_class": predicted_class,
        "predicted_duration": predicted_duration,
        "probabilities": probabilities.tolist()
    }
    if review_id is not None:
        response["review_id"] = review_id

    return jsonify(response)

if __name__ == '__main__':
    # By default, Flask runs on port 5000
    app.run(debug=True)

import torch
import numpy as np
from ai.src.time.train_time import (
    StudyDurationPredictor,
    map_class_to_duration,
    map_break_frequency_to_minutes,
    encode_study_intensity,
    load_and_preprocess_data
)

def load_model(model_path="study_duration_model.pth", input_dim=3):
    """ Load the trained PyTorch model for inference. """
    model = StudyDurationPredictor(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_study_duration(model, scaler, typical_study_str, break_freq_str, schedule_sat):
    """
    Predict tomorrow's recommended study duration.
    """
    study_intensity = encode_study_intensity(typical_study_str)
    break_freq_mins = map_break_frequency_to_minutes(break_freq_str)

    new_data = np.array([[study_intensity, break_freq_mins, schedule_sat]], dtype=np.float32)

    new_data_scaled = scaler.transform(new_data)
    new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(new_tensor)
        pred_class = torch.argmax(logits, dim=1).item()

    pred_label = map_class_to_duration(pred_class)
    return pred_class, pred_label

def main():
    # Load trained model
    model = load_model("study_duration_model.pth", input_dim=3)

    # Load scaler for correct preprocessing
    _, _, scaler, _ = load_and_preprocess_data()

    # 20 Test Cases
    test_cases = [
        ("Less than 30 minutes", "Rarely", 2),
        ("30-60 minutes", "Often", 4),
        ("1-2 hours", "Sometimes", 3),
        ("More than 2 hours", "Never", 5),
        ("1-2 hours", "Always", 1),
        ("30-60 minutes", "Rarely", 1),
        ("Less than 30 minutes", "Always", 3),
        ("More than 2 hours", "Sometimes", 2),
        ("1-2 hours", "Often", 5),
        ("30-60 minutes", "Never", 4),
        ("1-2 hours", "Rarely", 2),
        ("Less than 30 minutes", "Sometimes", 1),
        ("More than 2 hours", "Always", 3),
        ("30-60 minutes", "Often", 2),
        ("1-2 hours", "Sometimes", 4),
        ("Less than 30 minutes", "Never", 5),
        ("More than 2 hours", "Rarely", 3),
        ("1-2 hours", "Always", 2),
        ("30-60 minutes", "Sometimes", 1),
        ("More than 2 hours", "Often", 4)
    ]

    # Run test cases
    for i, (study, break_freq, sat) in enumerate(test_cases, 1):
        pred_class, pred_label = predict_study_duration(model, scaler, study, break_freq, sat)
        print(f"Test Case {i}: {study}, {break_freq}, {sat} -> Predicted: {pred_label}")

if __name__ == '__main__':
    main()

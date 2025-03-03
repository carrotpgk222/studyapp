import torch
import numpy as np
from time_model import TimeModel

def load_time_model(input_dim, model_path='time_model.pth'):
    model = TimeModel(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

if __name__ == '__main__':
    # Suppose we have the same 8 features for new user data
    input_dim = 8
    model = load_time_model(input_dim, 'time_model.pth')

    # Example new data (e.g., time_of_day=10, energy_level=4, prev_study=60, etc.)
    # We'll just use random for demonstration:
    new_data = np.random.rand(1, input_dim)

    # Convert to tensor
    new_data_tensor = torch.tensor(new_data, dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        study_pred, break_pred = model(new_data_tensor)
    
    # Convert predictions to Python floats
    recommended_study = float(study_pred[0].item())
    recommended_break = float(break_pred[0].item())
    
    print(f"Recommended study duration (minutes): {recommended_study:.2f}")
    print(f"Recommended break time (minutes): {recommended_break:.2f}")

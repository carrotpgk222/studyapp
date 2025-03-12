# predict.py
import torch
import pandas as pd
from train_model import ComplexSubjectPredictor 

# Load subject mapping
data = pd.read_csv("C:/Users/suban/studyapp-1/ai/src/subject_recommendation/train/study.csv")
subject_mapping = {subj: i for i, subj in enumerate(data["what subject are you doing today?"].unique())}
reverse_mapping = {i: subj for subj, i in subject_mapping.items()}

# Initialize model
num_subjects = len(subject_mapping)
model = ComplexSubjectPredictor(num_subjects)
model.load_state_dict(torch.load("complex_model.pth"))  # Ensure the path is correct
model.eval()

# Function to get valid user input
def get_valid_input(prompt, valid_range=(1, 5)):
    while True:
        try:
            value = float(input(prompt))
            if valid_range[0] <= value <= valid_range[1]:  # Ensuring value is within the valid range
                return value
            else:
                print(f"❌ Invalid input. Please enter a number between {valid_range[0]} and {valid_range[1]}.")
        except ValueError:
            print("❌ Invalid input. Please enter numerical values.")

# Get user input for subject, satisfaction, and difficulty
print("Subjects:")
for idx, subj in enumerate(reverse_mapping.values()):
    print(f"{idx}: {subj}")
    
subject_input = get_valid_input("Enter the subject number you're studying (e.g., 0 for Math):", (0, num_subjects-1))
satisfaction = get_valid_input("Enter satisfaction level (1-5): ")
difficulty = get_valid_input("Enter difficulty level (1-5): ")

# Map the user input subject to the corresponding numeric value
subject_label = subject_input

# Predict best subject based on the given inputs
input_tensor = torch.tensor([[satisfaction, difficulty]], dtype=torch.float32)
output = model(input_tensor)
predicted_subject = torch.argmax(output, dim=1).item()

# Get a recommendation based on the predicted subject
recommended_subject = reverse_mapping[predicted_subject]
print(f"📚 You are studying {reverse_mapping[subject_label]}, and based on your satisfaction and difficulty, we recommend: {recommended_subject}")

import torch
import torch.nn as nn
import pandas as pd

# Load subject mapping
data = pd.read_csv("../train/study.csv")
subject_mapping = {subj: i for i, subj in enumerate(data["what subject are you doing today?"].unique())}
reverse_mapping = {i: subj for subj, i in subject_mapping.items()}

# Load trained model
class SubjectPredictor(nn.Module):
    def __init__(self):
        super(SubjectPredictor, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, len(subject_mapping))  # Output: subjects

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SubjectPredictor()
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

# Get user input
satisfaction = float(input("Enter satisfaction level (1-5): "))
difficulty = float(input("Enter difficulty level (1-5): "))

# Predict best subject
input_tensor = torch.tensor([[satisfaction, difficulty]], dtype=torch.float32)
output = model(input_tensor)
predicted_subject = torch.argmax(output, dim=1).item()

print(f"📚 Recommended subject: {reverse_mapping[predicted_subject]}")

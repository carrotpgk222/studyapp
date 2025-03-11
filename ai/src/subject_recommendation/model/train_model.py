import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Load training data from CSV
data = pd.read_csv("C:/Users/bryan/studyapp/ai/src/subject_recommendation/train/study.csv")

# Check for NaN or Inf values and replace them
print("Checking NaN in dataset:", data.isna().sum())
print("Checking Inf in dataset:", (data == np.inf).sum())

# Replace NaN or Inf values with zeros
data = data.replace([np.inf, -np.inf], np.nan)  # Replace Inf with NaN
data = data.fillna(0)  # Replace NaN with 0, or you can use other imputation methods

# Convert subjects to numerical labels
subject_mapping = {subj: i for i, subj in enumerate(data["what subject are you doing today?"].unique())}
data["Subject"] = data["what subject are you doing today?"].map(subject_mapping)

# Prepare training input (X) and target output (y)
X = torch.tensor(data[["On a scale of 1 to 5 how satisfied your study session was", 
                       "On a scale of 1 (easy) to 5 (very challenging), how difficult do you find each subject?"]].values, dtype=torch.float32)
y = torch.tensor(data["Subject"].values, dtype=torch.long)

# Define a simple neural network
class SubjectPredictor(nn.Module):
    def __init__(self):
        super(SubjectPredictor, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # Input: satisfaction & difficulty
        self.fc2 = nn.Linear(8, len(subject_mapping))  # Output: subjects

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Train the model
model = SubjectPredictor()

# Criterion (Loss function) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    output = model(X)  # Forward pass
    loss = criterion(output, y)  # Compute loss

    if torch.isnan(loss).any():  # Check if loss is NaN
        print(f"Loss became NaN at epoch {epoch+1}, skipping this batch.")
        continue  # Skip this batch if loss is NaN

    loss.backward()  # Backward pass (compute gradients)
    optimizer.step()  # Update model parameters

    # Print loss at each epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

# Save trained model
torch.save(model.state_dict(), "model/model.pth")
print("✅ Model trained and saved!")

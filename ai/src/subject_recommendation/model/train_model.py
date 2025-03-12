# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load training data from CSV
data = pd.read_csv("C:/Users/suban/studyapp-1/ai/src/subject_recommendation/train/study.csv")

# Check for NaN or Inf values and replace them
print("Checking NaN in dataset:", data.isna().sum())
print("Checking Inf in dataset:", (data == np.inf).sum())

# Replace NaN or Inf values with zeros
data = data.replace([np.inf, -np.inf], np.nan)  # Replace Inf with NaN
data = data.fillna(0)  # Replace NaN with 0, or you can use other imputation methods

# Convert subjects to numerical labels
subject_mapping = {subj: i for i, subj in enumerate(data["what subject are you doing today?"].unique())}
data["Subject"] = data["what subject are you doing today?"].map(subject_mapping)

# Normalize satisfaction and difficulty (scale between 0 and 1)
data["Satisfaction"] = data["On a scale of 1 to 5 how satisfied your study session was"] / 5.0
data["Difficulty"] = data["On a scale of 1 (easy) to 5 (very challenging), how difficult do you find each subject?"] / 5.0

# Prepare training input (X) and target output (y)
X = torch.tensor(data[["Satisfaction", "Difficulty"]].values, dtype=torch.float32)
y = torch.tensor(data["Subject"].values, dtype=torch.long)

# Define a more complex neural network
class ComplexSubjectPredictor(nn.Module):
    def __init__(self, num_subjects):
        super(ComplexSubjectPredictor, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input: satisfaction & difficulty, increased neurons
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization for first layer
        self.fc2 = nn.Linear(64, 128)  # Hidden layer with more neurons
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization for second layer
        self.fc3 = nn.Linear(128, 64)  # Third layer
        self.fc4 = nn.Linear(64, num_subjects)  # Output layer
        
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))  # Forward pass with batch normalization and ReLU
        x = self.dropout(x)  # Apply dropout
        x = torch.relu(self.bn2(self.fc2(x)))  # Forward pass with batch normalization and ReLU
        x = self.dropout(x)  # Apply dropout
        x = torch.relu(self.fc3(x))  # ReLU activation for the third layer
        return self.fc4(x)  # Output layer without activation (CrossEntropyLoss handles it)

# Initialize model
num_subjects = len(subject_mapping)
model = ComplexSubjectPredictor(num_subjects)

# Criterion (Loss function) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.008)  # Optimizer with learning rate scheduler

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Reduce LR every 50 epochs

# Training loop (with lower epochs and handling NaN loss)
epochs = 1000  # Low epoch count for quick training
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    output = model(X)  # Forward pass
    loss = criterion(output, y)  # Compute loss

    if torch.isnan(loss).any():  # Check if loss is NaN
        print(f"Loss became NaN at epoch {epoch+1}, skipping this batch.")
        continue  # Skip this batch if loss is NaN

    loss.backward()  # Backward pass (compute gradients)
    optimizer.step()  # Update model parameters

    # Adjust learning rate
    scheduler.step()

    # Print loss at each epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

# Save trained model
torch.save(model.state_dict(), "complex_model.pth")
print("✅ Complex model trained and saved!")

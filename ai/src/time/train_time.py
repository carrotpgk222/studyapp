import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle  # <-- for saving the scaler

#####################################
# 1) MAPPING FUNCTIONS
#####################################

def map_duration_to_class(duration_str):
    mapping = {
        'Less than 30 minutes': 0,
        '30-60 minutes': 1,
        '1-2 hours': 2,
        'More than 2 hours': 3
    }
    return mapping.get(duration_str, -1)

def map_class_to_duration(cls):
    rev_mapping = {
        0: 'Less than 30 minutes',
        1: '30-60 minutes',
        2: '1-2 hours',
        3: 'More than 2 hours'
    }
    return rev_mapping.get(cls, 'Unknown')

def map_break_frequency_to_minutes(freq_str):
    mapping = {
        'Never': 0,
        'Rarely': 5,
        'Sometimes': 7,
        'Often': 10,
        'Always': 15
    }
    return mapping.get(freq_str, -1)

def encode_study_intensity(duration_str):
    cls = map_duration_to_class(duration_str)
    return cls / 3.0 if cls != -1 else 0.0  

#####################################
# 2) IMPROVED MODEL DEFINITION
#####################################

class StudyDurationPredictor(nn.Module):
    def __init__(self, input_dim=3):
        super(StudyDurationPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.3)  # Reduced dropout
        self.out = nn.Linear(16, 4)  

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        logits = self.out(x)
        return logits

#####################################
# 3) DATA PREPROCESSING
#####################################

def load_and_preprocess_data():
    df = pd.read_csv("C:/Users/bryan/studyapp/ai/src/time/time_data_modified.csv")

    required_cols = [
        "Typical Study Session Duration",
        "Break Frequency",
        "Schedule Satisfaction",
        "Tomorrow Study Time"
    ]
    df.dropna(subset=required_cols, inplace=True)

    df["Study_Intensity"] = df["Typical Study Session Duration"].apply(encode_study_intensity)
    df["BreakFreqMins"] = df["Break Frequency"].apply(map_break_frequency_to_minutes)
    df = df[df["BreakFreqMins"] != -1]
    df["Schedule Satisfaction"] = pd.to_numeric(df["Schedule Satisfaction"], errors='coerce')
    df.dropna(subset=["Schedule Satisfaction"], inplace=True)

    df["Target_Class"] = df["Tomorrow Study Time"].apply(map_duration_to_class)
    df = df[df["Target_Class"] != -1]

    feature_cols = ["Study_Intensity", "BreakFreqMins", "Schedule Satisfaction"]
    X = df[feature_cols].values  
    y = df["Target_Class"].values  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor, scaler, feature_cols

#####################################
# 4) TRAINING LOOP
#####################################

def train_model(model, optimizer, X, y, epochs=500):
    class_weights = torch.tensor([1.1, 1.0, 1.2, 1.4], dtype=torch.float32)  
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(X.size(0))
        X_epoch = X[perm]
        y_epoch = y[perm]

        logits = model(X_epoch)
        loss = criterion(logits, y_epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

#####################################
# 5) TRAINING SCRIPT
#####################################

def main():
    X_tensor, y_tensor, scaler, feature_cols = load_and_preprocess_data()
    input_dim = X_tensor.shape[1]  
    model = StudyDurationPredictor(input_dim=3)  
    optimizer = optim.Adam(model.parameters(), lr=0.0008)  # Lowered LR for stability

    print("Training model...")
    train_model(model, optimizer, X_tensor, y_tensor, epochs=500)

    # Save the model weights
    torch.save(model.state_dict(), "study_duration_model.pth")

    # Save the scaler to a pickle file
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Training complete. Model and scaler saved.")

if __name__ == '__main__':
    main()

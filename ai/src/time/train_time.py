import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle  # For saving the scaler

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

# New mapping functions to convert text to numeric hours
def map_study_duration_to_hours(text):
    # Adjust these values as needed (in hours)
    mapping = {
        'Less than 30 minutes': 0.5,   # e.g. 30 minutes approximated as 0.5 hours
        '30-60 minutes': 0.75,          # average 45 minutes = 0.75 hour
        '1-2 hours': 1.5,
        'More than 2 hours': 2.5
    }
    return mapping.get(text, np.nan)

def map_break_frequency_to_hours(text):
    # Assuming break frequency is provided as a text category:
    mapping = {
        'Never': 0.0,
        'Rarely': 5/60.0,      # 5 minutes ≈ 0.0833 hours
        'Sometimes': 7/60.0,   # 7 minutes ≈ 0.1167 hours
        'Often': 10/60.0,      # 10 minutes ≈ 0.1667 hours
        'Always': 15/60.0      # 15 minutes = 0.25 hours
    }
    return mapping.get(text, np.nan)

#####################################
# 2) MODEL DEFINITION
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
        self.dropout = nn.Dropout(0.3)  
        self.out = nn.Linear(16, 4)  

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.out(x)

#####################################
# 3) DATA PREPROCESSING (USING NUMERIC HOURS)
#####################################

def load_and_preprocess_data():
    df = pd.read_csv("C:/Users/bryan/studyapp/ai/src/time/time_data_modified.csv")
    
    print("Initial Columns:", df.columns.tolist())
    print("Initial DataFrame shape:", df.shape)
    print("First few rows:\n", df.head())
    
    # Drop rows missing required columns
    required_cols = [
        "Typical Study Session Duration",
        "Break Frequency",
        "Schedule Satisfaction",
        "Tomorrow Study Time"
    ]
    df.dropna(subset=required_cols, inplace=True)
    print("After dropna on required columns, shape:", df.shape)
    
    # Map textual values to numeric hours
    df["Typical Study Session Duration Numeric"] = df["Typical Study Session Duration"].apply(map_study_duration_to_hours)
    df["Break Frequency Numeric"] = df["Break Frequency"].apply(map_break_frequency_to_hours)
    df["Schedule Satisfaction"] = pd.to_numeric(df["Schedule Satisfaction"], errors='coerce')
    
    print("After mapping, sample data:")
    print(df[["Typical Study Session Duration", "Typical Study Session Duration Numeric", 
              "Break Frequency", "Break Frequency Numeric", "Schedule Satisfaction"]].head())
    
    # Drop rows where mapping produced NaN values
    df.dropna(subset=["Typical Study Session Duration Numeric", "Break Frequency Numeric", "Schedule Satisfaction"], inplace=True)
    print("After dropna post mapping, shape:", df.shape)
    
    # Map target variable using the original strings in "Tomorrow Study Time"
    df["Target_Class"] = df["Tomorrow Study Time"].apply(map_duration_to_class)
    print("Unique Target_Class values (before filtering):", df["Target_Class"].unique())
    df = df[df["Target_Class"] != -1]
    print("After filtering invalid Target_Class, shape:", df.shape)
    
    # Features now use the mapped numeric columns
    feature_cols = ["Typical Study Session Duration Numeric", "Break Frequency Numeric", "Schedule Satisfaction"]
    X = df[feature_cols].values  
    y = df["Target_Class"].values  
    print("Features shape:", X.shape, "Target shape:", y.shape)
    
    if X.shape[0] == 0:
        raise ValueError("Error: No valid data left after preprocessing. Check dataset or input columns.")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Scaled features sample:\n", X_scaled[:5])
    
    # Convert to PyTorch tensors
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
        X_epoch, y_epoch = X[perm], y[perm]
        
        optimizer.zero_grad()
        logits = model(X_epoch)
        loss = criterion(logits, y_epoch)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

#####################################
# 5) TRAINING SCRIPT
#####################################

def main():
    X_tensor, y_tensor, scaler, feature_cols = load_and_preprocess_data()
    model = StudyDurationPredictor(input_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    
    print("Training model...")
    train_model(model, optimizer, X_tensor, y_tensor, epochs=500)
    
    # Save the model weights
    torch.save(model.state_dict(), "study_duration_model.pth")
    print("Model weights saved.")
    
    # Save the scaler to a pickle file
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved.")
    
    print("Training complete. Model and scaler saved.")

if __name__ == '__main__':
    main()

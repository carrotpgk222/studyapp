import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TimeModel(nn.Module):
    def __init__(self, input_dim):
        super(TimeModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.study_duration = nn.Linear(16, 1)
        self.break_time = nn.Linear(16, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        study = self.study_duration(x)  # Predicted study duration
        breakt = self.break_time(x)     # Predicted break time
        return study, breakt

def train_time_model(model, optimizer, criterion, X, y, epochs=10):
    """
    Trains the time management model.
    :param model: PyTorch model (TimeModel).
    :param optimizer: PyTorch optimizer (Adam, SGD, etc.).
    :param criterion: Loss function (MSELoss).
    :param X: Feature tensor of shape [num_samples, input_dim].
    :param y: Target tensor of shape [num_samples, 2].
    :param epochs: Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        # Shuffle the data each epoch (optional)
        perm = torch.randperm(X.size(0))
        X = X[perm]
        y = y[perm]

        # Forward pass
        pred_study, pred_break = model(X)
        
        # y[:, 0] -> study_duration, y[:, 1] -> break_time
        # Reshape predictions to match y's shape
        pred_study = pred_study.view(-1)
        pred_break = pred_break.view(-1)
        
        # Calculate MSE for both outputs
        loss_study = criterion(pred_study, y[:, 0])
        loss_break = criterion(pred_break, y[:, 1])
        loss = loss_study + loss_break
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def main():
    # -----------------------------
    # 1. Load or Generate Data
    # -----------------------------
    # Example: Generating synthetic data for demonstration.
    # In practice, replace this with reading from CSV or database.
    num_samples = 500
    # Suppose we have 8 features (time_of_day, energy_level, etc.)
    input_dim = 8
    
    # Synthetic features
    X_np = np.random.rand(num_samples, input_dim)
    
    # Synthetic targets:
    # study_duration in [30, 180], break_time in [5, 30]
    y_study = np.random.rand(num_samples) * (180 - 30) + 30
    y_break = np.random.rand(num_samples) * (30 - 5) + 5
    y_np = np.column_stack((y_study, y_break))

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    # -----------------------------
    # 2. Instantiate Model & Trainer
    # -----------------------------
    model = TimeModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # -----------------------------
    # 3. Train the Model
    # -----------------------------
    train_time_model(model, optimizer, criterion, X_tensor, y_tensor, epochs=10)
    
    # -----------------------------
    # 4. Save the Model
    # -----------------------------
    torch.save(model.state_dict(), 'time_model.pth')

if __name__ == '__main__':
    main()
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Load the data from Excel
file_path = "data/survey_data.xlsx"
df = pd.read_excel(file_path, sheet_name="SessionFeedback")

# Map textual feedback to numerical values (you can adjust this mapping as needed)
feedback_mapping = {"Productive": 1, "Tired": 2, "Overwhelmed": 3}
df["User Feedback"] = df["User Feedback"].map(feedback_mapping)

# Features: Well-being score, Time spent, User feedback, Performance score
X = df[["Well-being Score", "Time Spent (min)", "User Feedback", "Performance Score"]].values.astype(np.float32)

# Target: Index or other labels (you can customize this for your use case)
y = np.array(df.index)  # Example: Using row index as a placeholder

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Define the model
class FeedbackModel(nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 4 input features
        self.fc2 = nn.Linear(16, 8)  # Hidden layer
        self.fc3 = nn.Linear(8, 3)   # 3 classes (feedback categories)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Initialize the model, loss function, and optimizer
model = FeedbackModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), "models/feedback_model.pth")

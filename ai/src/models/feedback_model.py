import torch
import numpy as np
import torch.nn as nn

# Load the trained model
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

# Load the trained model from the file
model = FeedbackModel()
model.load_state_dict(torch.load("models/feedback_model.pth"))
model.eval()

# Sample new input (replace with actual user input)
new_input = np.array([[6, 25, 2, 75]])  # Example: Well-being 6, Time 25 mins, Feedback "Tired", Performance 75%

# Convert input to PyTorch tensor
input_tensor = torch.tensor(new_input, dtype=torch.float32)

# Get prediction from the model
output = model(input_tensor)

# Get the predicted class (feedback type)
predicted_class = torch.argmax(output, dim=1).item()

# Feedback options (based on training data)
feedback_options = ["Excellent! Keep up the momentum.", "Try reducing session time for better focus.", "Consider a break or a different learning approach."]

print("Predicted Feedback:", feedback_options[predicted_class])

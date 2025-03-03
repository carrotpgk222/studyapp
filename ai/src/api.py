from flask import Flask, request, jsonify
import torch
import numpy as np

app = Flask(__name__)

# Define model (same as before)
class FeedbackModel(nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Load the trained model
model = FeedbackModel()
model.load_state_dict(torch.load("models/feedback_model.pth"))
model.eval()

# API route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get data from POST request
    data = request.get_json()

    # Extract input features (assuming they are in the JSON format)
    input_data = np.array([data["wellbeing_score"], data["time_spent"], data["feedback"], data["performance_score"]])
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Get prediction
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    # Define feedback messages
    feedback_options = ["Excellent! Keep up the momentum.", "Try reducing session time for better focus.", "Consider a break or a different learning approach."]

    # Return prediction as JSON response
    return jsonify({"feedback": feedback_options[predicted_class]})

if __name__ == "__main__":
    app.run(debug=True)

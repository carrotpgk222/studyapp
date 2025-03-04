import torch
import torch.nn as nn

class SubjectModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SubjectModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)  # logits output; softmax applied externally
        return x
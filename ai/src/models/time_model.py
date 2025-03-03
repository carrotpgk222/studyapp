import torch
import torch.nn as nn

class TimeModel(nn.Module):
    def __init__(self, input_dim):
        super(TimeModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.study_duration = nn.Linear(16, 1)
        self.break_time = nn.Linear(16, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        study = self.study_duration(x)
        breakt = self.break_time(x)
        return study, breakt

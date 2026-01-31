import torch.nn as nn
import torch.nn.functional as F


# 4. Define the model architecture with regularization
class ConeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 128)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

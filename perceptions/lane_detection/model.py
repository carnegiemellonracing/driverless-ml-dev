import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn.functional as F
from data_loader import *

# 4. Define the model architecture with regularization
class ConeClassifier(nn.Module):
    def __init__(self):
        super(ConeClassifier, self).__init__()
        # Fully connected layers with reduced complexity
        self.fc1 = nn.Linear(8, 100)#, bias=False
        self.fc2 = nn.Linear(100, 1)
        
    def forward(self, x):
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
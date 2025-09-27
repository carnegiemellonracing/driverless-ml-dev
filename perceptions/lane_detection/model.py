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
        # Convolutional layers with reduced complexity
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Adaptive pooling to get fixed size output regardless of input length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)  # Fixed output size
        
        # Calculate the flattened size after conv layers and pooling
        self._get_conv_output = lambda x: 32 * 16  # channels * fixed_length
        
        # Fully connected layers with reduced complexity
        self.fc1 = nn.Linear(self._get_conv_output(None), 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Ensure input is 3D: [batch_size, sequence_length, channels]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Change to [batch, channels, sequence_length]
        x = x.permute(0, 2, 1)  # [batch, 2, points]
        
        # Convolutional layers with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Adaptive pooling to get fixed size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def compute_iou(pred, gt):
    """Compute IoU between predicted and ground truth points"""
    # Convert to sets of points for IoU calculation
    pred_set = set(map(tuple, pred.reshape(-1, 2)))
    gt_set = set(map(tuple, gt.reshape(-1, 2)))
    
    # Calculate intersection and union
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    
    return intersection / max(union, 1)  # Avoid division by zero

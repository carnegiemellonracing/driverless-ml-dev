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

def compute_iou(pred, gt):
    """Compute IoU between predicted and ground truth points"""
    # Convert to sets of points for IoU calculation
    pred_set = set(map(tuple, pred.reshape(-1, 2)))
    gt_set = set(map(tuple, gt.reshape(-1, 2)))
    
    # Calculate intersection and union
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    
    return intersection / max(union, 1)  # Avoid division by zero

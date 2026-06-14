import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util import chamfer_distance

# default hyperparameters
NUM_OUTPUT_POINTS=1024
GLOBAL_FEAT_DIM=1024

class FCDecoder(nn.Module):
    def __init__(self, num_points=NUM_OUTPUT_POINTS, global_feat_dim=GLOBAL_FEAT_DIM):
        super(FCDecoder, self).__init__()

        self.global_feat_dim = global_feat_dim

        self.fc1 = nn.Linear(global_feat_dim, global_feat_dim)
        self.fc2 = nn.Linear(global_feat_dim, global_feat_dim)
        self.fc3 = nn.Linear(global_feat_dim, num_points * 3)

    def forward(self, global_feature):
        x = F.relu(self.fc1(global_feature))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, NUM_OUTPUT_POINTS, 3)
        return x
    
def fc_decoder_loss(pred, gt):
    return chamfer_distance(pred, gt)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mock_global_feat = torch.randn(4, 1024).to(device)
    
    decoder = FCDecoder().to(device)
    out_coarse = decoder(mock_global_feat)
    
    print(f"Input Feature Shape:  {mock_global_feat.shape}")
    print(f"Output Coarse Shape: {out_coarse.shape} (Expected: [4, 1024, 3])")
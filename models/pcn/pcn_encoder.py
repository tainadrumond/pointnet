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

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat_dim=1024):
        super(PointNetEncoder, self).__init__()
        self.global_feat_dim = global_feat_dim

        self.conv1 = nn.Conv1d(3, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(512, 512, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, global_feat_dim, 1)
        self.bn4 = nn.BatchNorm1d(global_feat_dim)
        

    def forward(self, point_cloud):
        B, N, C = point_cloud.shape # (B, N, 3)
        point_cloud_trans = point_cloud.transpose(1, 2) # (B, 3, N)

        # 1st shared mlp
        feat1 = F.relu(self.bn1(self.conv1(point_cloud_trans)))
        feat1 = F.relu(self.bn2(self.conv2(feat1))) # (B, 256, N)

        # 1st point-wise maxpool
        global1 = torch.max(feat1, 2, keepdim=True)[0] # (B, 256, 1)

        # expand
        global1_expanded = global1.repeat(1, 1, N) # (B, 256, N)
        fused = torch.cat([feat1, global1_expanded], dim=1) # (B, 512, N)

        # 2nd shared mlp
        feat2 = F.relu(self.bn3(self.conv3(fused)))
        feat2 = F.relu(self.bn4(self.conv4(fused))) # (B, global_feat_dim, N)

        # 2nd point-wise maxpool
        global_feature = torch.max(feat2, 2)[0] # (B, global_feat_dim)

        return global_feature


if __name__ == '__main__':
    inputs = torch.zeros((32, 1024, 3))
    
    model = PointNetEncoder()
    model.train()
    
    output = model(inputs)
    print(output)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

class input_transform_net(nn.Module):
    """ Input (XYZ) Transform Net """
    def __init__(self, K=3):
        super(input_transform_net, self).__init__()
        self.K = K
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * K)
        
        # Inicializa pesos com 0 e bias com a matriz identidade
        nn.init.constant_(self.transform.weight, 0.0)
        nn.init.constant_(self.transform.bias, 0.0)
        self.transform.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32))

    def forward(self, point_cloud):
        # Transpõe (B, N, 3) para (B, 3, N) para a convolução 1D
        x = point_cloud.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        x = self.transform(x)
        x = x.view(-1, 3, self.K)
        
        return x


class feature_transform_net(nn.Module):
    """ Feature Transform Net """
    def __init__(self, K=64):
        super(feature_transform_net, self).__init__()
        self.K = K
        
        self.conv1 = nn.Conv1d(K, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, K * K)
        
        # Inicializa pesos com 0 e bias com a matriz identidade
        nn.init.constant_(self.transform.weight, 0.0)
        nn.init.constant_(self.transform.bias, 0.0)
        self.transform.bias.data.copy_(torch.eye(K).view(-1))

    def forward(self, inputs):
        # A entrada já chega como (B, K, N) da PointNet principal
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        x = self.transform(x)
        x = x.view(-1, self.K, self.K)
        
        return x
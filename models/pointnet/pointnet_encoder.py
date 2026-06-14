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

from models.pointnet.transform_nets import input_transform_net, feature_transform_net

GLOBAL_FEAT_DIM=1024

class PointNetEncoder(nn.Module):
    """ Conversão direta da sua função get_model para PyTorch """
    def __init__(self, global_feat_dim=GLOBAL_FEAT_DIM):
        super(PointNetEncoder, self).__init__()
        self.global_feat_dim = global_feat_dim
        
        self.input_transform = input_transform_net(K=3)
        self.feature_transform = feature_transform_net(K=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, self.global_feat_dim, 1)
        self.bn5 = nn.BatchNorm1d(self.global_feat_dim)

        #self.fc1 = nn.Linear(1024, 512)
        #self.bn6 = nn.BatchNorm1d(512)
        #self.dropout1 = nn.Dropout(p=0.3) 
        
        #self.fc2 = nn.Linear(512, 256)
        #self.bn7 = nn.BatchNorm1d(256)
        #self.dropout2 = nn.Dropout(p=0.3)
        

    def forward(self, point_cloud):
        end_points = {}

        # --- transform_net1 ---
        transform = self.input_transform(point_cloud) 
        point_cloud_transformed = torch.bmm(point_cloud, transform)

        input_image = point_cloud_transformed.transpose(1, 2) # (B, 3, N)

        net = F.relu(self.bn1(self.conv1(input_image)))
        net = F.relu(self.bn2(self.conv2(net)))

        # --- transform_net2 ---
        transform2 = self.feature_transform(net)
        end_points['transform'] = transform2
        
        net_transformed = torch.bmm(net.transpose(1, 2), transform2)
        net_transformed = net_transformed.transpose(1, 2) # Retorna para (B, 64, N)

        net = F.relu(self.bn3(self.conv3(net_transformed)))
        net = F.relu(self.bn4(self.conv4(net)))
        net = F.relu(self.bn5(self.conv5(net)))

        # Symmetric function: max pooling
        net = torch.max(net, 2, keepdim=True)[0]
        global_feature = net.view(-1, self.global_feat_dim) # (B, global_feat_dim)

        # net = F.relu(self.bn6(self.fc1(net)))
        # net = self.dropout1(net)
        # net = F.relu(self.bn7(self.fc2(net)))
        # net = self.dropout2(net)
        # net = self.fc3(net)

        return global_feature, end_points


if __name__ == '__main__':
    inputs = torch.zeros((32, 1024, 3))
    
    model = PointNetEncoder()
    model.train() # is_training = True
    
    outputs, _ = model(inputs)
    print(outputs)
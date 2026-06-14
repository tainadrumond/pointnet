import os
import sys
import torch
import torch.nn as nn


# path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.pointnet.pointnet_encoder import PointNetEncoder
from models.pcn.fc_decoder import FCDecoder
from models.pcn.folding_based_decoder import FoldingBasedDecoder
from util import chamfer_distance


# default hyperparameters
NUM_COARSE_POINTS = 1024
GLOBAL_FEAT_DIM = 1024
GRID_SIZE = 4
GRID_SCALE = 0.05

class PCN(nn.Module):
    def __init__(self, num_coarse=NUM_COARSE_POINTS, global_feat_dim=GLOBAL_FEAT_DIM, grid_size=GRID_SIZE, grid_scale=GRID_SCALE):
        super(PCN, self).__init__()
        
        self.encoder = PointNetEncoder(global_feat_dim=global_feat_dim)
        
        self.coarse_decoder = FCDecoder(num_points=num_coarse, global_feat_dim=global_feat_dim)
        
        self.dense_decoder = FoldingBasedDecoder(num_coarse_points=num_coarse, global_feat_dim=global_feat_dim, grid_size=grid_size, grid_scale=grid_scale)

    def forward(self, x):
        global_features, _ = self.encoder(x)
        coarse = self.coarse_decoder(global_features)
        dense = self.dense_decoder(global_features, coarse)

        return coarse, dense

def pcn_loss(pred_coarse, pred_detail, gt, alpha=1.0):
    loss_coarse = chamfer_distance(pred_coarse, gt)
    loss_detail = chamfer_distance(pred_detail, gt)
    return (alpha * loss_coarse) + loss_detail


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mock_partial_input = torch.randn(4, 2048, 3).to(device)
    mock_gt = torch.randn(4, 16384, 3).to(device)
    
    model = PCN().to(device)
    coarse_out, dense_out = model(mock_partial_input)
    
    print(f"Partial Input Shape: {mock_partial_input.shape}")
    print(f"Output Coarse Shape: {coarse_out.shape} (Expected: [4, 1024, 3])")
    print(f"Output Dense Shape:  {dense_out.shape} (Expected: [4, 16384, 3])")
    
    loss = pcn_loss(coarse_out, dense_out, mock_gt)
    print(f"Calculated Joint PCN Loss Value: {loss.item()}")
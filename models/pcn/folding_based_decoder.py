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
NUM_COARSE_POINTS=1024
GLOBAL_FEAT_DIM=1024
GRID_SIZE=4
GRID_SCALE=0.5

class FoldingBasedDecoder(nn.Module):
    def __init__(self, num_coarse_points=NUM_COARSE_POINTS, global_feat_dim=GLOBAL_FEAT_DIM, grid_size=GRID_SIZE, grid_scale=GRID_SCALE):
        super(FoldingBasedDecoder, self).__init__()

        self.num_coarse_points = num_coarse_points
        self.global_feat_dim = global_feat_dim
        self.grid_size = grid_size
        self.grid_scale = grid_scale

        self.num_fine_points = self.num_coarse_points * (self.grid_size ** 2)

        # 2 (grid) + 3 (coarse point dim) + global_feat_dim
        in_channels = 2 + 3 + global_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, global_feature, coarse_points):
        B = global_feature.shape[0]
        
        # 1. Generate the standard local 2D grid
        x = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size, device=global_feature.device)
        y = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size, device=global_feature.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        
        grid = torch.stack([grid_x, grid_y], dim=2)                     # Shape: (grid_size, grid_size, 2)
        grid = grid.view(-1, 2).unsqueeze(0)                            # Shape: (1, grid_size**2, 2)
        grid_feat = grid.repeat(B, self.num_coarse_points, 1)                  # Shape: (B, num_fine, 2)

        # 2. Tile the coarse point coordinates
        point_feat = coarse_points.unsqueeze(2).repeat(1, 1, self.grid_size ** 2, 1)
        point_feat = point_feat.view(B, self.num_fine_points, 3)               # Shape: (B, num_fine, 3)

        # 3. Tile the global feature vector
        global_feat = global_feature.unsqueeze(1).repeat(1, self.num_fine_points, 1) # Shape: (B, num_fine, global_feat_dim)

        # 4. Concatenate features along the channel axis (dim=2)
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)   # Shape: (B, num_fine, in_channels)

        # 5. Get the centers for the final relative shift (same as point_feat mapping)
        center = coarse_points.unsqueeze(2).repeat(1, 1, self.grid_size ** 2, 1)
        center = center.view(B, self.num_fine_points, 3)                       # Shape: (B, num_fine, 3)

        # 6. Run folding operation and translate patches into global coordinate space
        fine = self.mlp(feat) + center                                  # Shape: (B, num_fine, 3)

        return fine
    
def folding_based_decoder_loss(pred_fine, gt):
    return chamfer_distance(pred_fine, gt)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock inputs: Batch size 4, Global Feature Dim 1024, Coarse points (4, 1024, 3)
    mock_global_feat = torch.randn(4, 1024).to(device)
    mock_coarse_points = torch.randn(4, 1024, 3).to(device)
    
    decoder = FoldingBasedDecoder().to(device)
    out_fine = decoder(mock_global_feat, mock_coarse_points)
    
    print(f"Input Coarse Shape: {mock_coarse_points.shape}")
    print(f"Output Fine Shape:  {out_fine.shape} (Expected: [4, 16384, 3])")
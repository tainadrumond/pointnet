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


class FoldingBasedDecoder(nn.Module):
    def __init__(self, num_coarse_points=1024, global_feat_dim=1024, grid_size=4, grid_scale=0.5):
        super(FoldingBasedDecoder, self).__init__()

        self.num_coarse = num_coarse_points
        self.global_feat_dim = global_feat_dim
        self.grid_size = grid_size
        self.grid_scale = grid_scale

        self.num_fine = self.num_coarse * (self.grid_size ** 2)

        # in_channels = 2 (grid) + 3 (coarse point dim) + global_feat_dim
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
        
        # generate 2d grid
        x = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size, device=global_feature.device)
        y = torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size, device=global_feature.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1) # (grid_size**2, 2)
        grid_feat = grid.repeat(self.num_coarse, 1).unsqueeze(0).repeat(B, 1, 1) # (B, num_fine, 2)

        # tile coarse points
        point_feat = coarse_points.unsqueeze(2).repeat(1, 1, self.grid_size**2, 1)
        point_feat = point_feat.reshape(B, self.num_fine, 3) # (B, num_fine, 3)

        # tile global feature vector
        global_feat = global_feature.unsqueeze(1).repeat(1, self.num_fine, 1) # (B, num_fine, global_feat_dim)

        # concat
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2) # (B, num_fine, 2 + 3 + global_feat_dim)

        # shared mlp
        fine = self.mlp(feat) + point_feat # (B, num_fine, 3)

        return fine


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mock_global_feat = torch.randn(4, 1024).to(device)
    mock_coarse_points = torch.randn(4, 1024, 3).to(device)
    
    decoder = FoldingBasedDecoder().to(device)
    out_fine = decoder(mock_global_feat, mock_coarse_points)
    
    print(f"Input Coarse Shape: {mock_coarse_points.shape}")
    print(f"Output Fine Shape:  {out_fine.shape} (Expected: [4, 16384, 3])")
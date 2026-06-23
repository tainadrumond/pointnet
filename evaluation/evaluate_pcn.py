import os
import sys
import json
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from scripts.pcn_provider import ShapeNetPLYDataset
from models.pcn.pcn import PCN, pcn_loss
from util import chamfer_distance

# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="config JSON filename (e.g. pcn or pcn_reduced)"
)
parser.add_argument(
    "--run",
    type=str,
    required=True,
    help="run directory name (e.g. 20260616_104256)"
)
parser.add_argument(
    "--epoch",
    type=int,
    default=None,
    help="epoch number to evaluate (defaults to the latest available)"
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="limit number of samples to evaluate for speed (e.g. 100)"
)
args = parser.parse_args()

# Load settings
config_path = os.path.join(BASE_DIR, f'configs/{args.config}.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Read settings
SEED = config['seed']
NUM_COARSE = config['num_coarse_points']
GLOBAL_FEAT_DIM = config['global_feat_dim']
GRID_SIZE = config['grid_size']
GRID_SCALE = config['grid_scale']

# Seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = PCN(
    num_coarse=NUM_COARSE,
    global_feat_dim=GLOBAL_FEAT_DIM,
    grid_size=GRID_SIZE,
    grid_scale=GRID_SCALE
).to(device)

# Find checkpoint
log_dir = os.path.join(BASE_DIR, "log", args.run)
if not os.path.exists(log_dir):
    raise FileNotFoundError(f"Run directory not found: {log_dir}")

checkpoint_path = None
selected_epoch = None
if args.epoch is not None:
    checkpoint_path = os.path.join(log_dir, f"pcn_epoch_{args.epoch}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint for epoch {args.epoch} not found at {checkpoint_path}")
    selected_epoch = args.epoch
else:
    # Look for the checkpoint with the highest epoch
    pth_files = glob.glob(os.path.join(log_dir, "pcn_epoch_*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No checkpoint files pcn_epoch_*.pth found in {log_dir}")
    
    epochs = []
    for f in pth_files:
        match = re.search(r"pcn_epoch_(\d+)\.pth", f)
        if match:
            epochs.append((int(match.group(1)), f))
    
    if not epochs:
        raise ValueError(f"Could not parse epoch numbers from checkpoint files in {log_dir}")
    
    # Sort by epoch descending
    epochs.sort(reverse=True, key=lambda x: x[0])
    selected_epoch, checkpoint_path = epochs[0]
    print(f"Automatically selected epoch {selected_epoch} checkpoint: {checkpoint_path}")

print(f"Loading weights from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Check dataset path
dataset_dir = os.path.join(BASE_DIR, "data/shapenet")
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")

# Load test dataset
base_dataset = ShapeNetPLYDataset(data_dir=dataset_dir, split="test", num_views=1)

if args.limit is not None and args.limit > 0:
    print(f"Limiting evaluation to a subset of {args.limit} samples out of {len(base_dataset)}.")
    # Use deterministic shuffle with seed for reproducibility
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(base_dataset), size=min(args.limit, len(base_dataset)), replace=False)
    test_dataset = Subset(base_dataset, indices)
    # Copy required attributes
    test_dataset.class_to_idx = base_dataset.class_to_idx
    test_dataset.num_classes = base_dataset.num_classes
else:
    test_dataset = base_dataset

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# Evaluate
def evaluate(model, loader):
    total_loss = 0.0
    total_coarse_cd = 0.0
    total_dense_cd = 0.0
    total_samples = 0
    
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    num_classes = len(test_dataset.class_to_idx)
    class_loss_sum = [0.0] * num_classes
    class_coarse_cd_sum = [0.0] * num_classes
    class_dense_cd_sum = [0.0] * num_classes
    class_samples_count = [0] * num_classes
    
    alpha = 1.0
    
    with torch.no_grad():
        for partial, complete, label in loader:
            batch_size = partial.shape[0]
            partial = partial.to(device)
            complete = complete.to(device)
            label = label.to(device)
            
            coarse_pred, dense_pred = model(partial)
            
            for i in range(batch_size):
                p_i = partial[i:i+1]
                c_i = complete[i:i+1]
                lbl_i = label[i].item()
                
                coarse_p_i = coarse_pred[i:i+1]
                dense_p_i = dense_pred[i:i+1]
                
                cd_coarse_i = chamfer_distance(coarse_p_i, c_i).item()
                cd_dense_i = chamfer_distance(dense_p_i, c_i).item()
                loss_i = (alpha * cd_coarse_i) + cd_dense_i
                
                total_loss += loss_i
                total_coarse_cd += cd_coarse_i
                total_dense_cd += cd_dense_i
                total_samples += 1
                
                class_loss_sum[lbl_i] += loss_i
                class_coarse_cd_sum[lbl_i] += cd_coarse_i
                class_dense_cd_sum[lbl_i] += cd_dense_i
                class_samples_count[lbl_i] += 1

    eval_loss = total_loss / total_samples
    eval_coarse_cd = total_coarse_cd / total_samples
    eval_dense_cd = total_dense_cd / total_samples
    
    class_results = {}
    for idx in range(num_classes):
        class_name = idx_to_class[idx]
        count = class_samples_count[idx]
        if count > 0:
            class_results[class_name] = {
                "count": count,
                "loss": class_loss_sum[idx] / count,
                "coarse_cd": class_coarse_cd_sum[idx] / count,
                "dense_cd": class_dense_cd_sum[idx] / count
            }
        else:
            class_results[class_name] = {
                "count": 0,
                "loss": 0.0,
                "coarse_cd": 0.0,
                "dense_cd": 0.0
            }
            
    valid_classes = [c for c in class_results.values() if c["count"] > 0]
    macro_avg_loss = np.mean([c["loss"] for c in valid_classes]) if valid_classes else 0.0
    macro_avg_coarse_cd = np.mean([c["coarse_cd"] for c in valid_classes]) if valid_classes else 0.0
    macro_avg_dense_cd = np.mean([c["dense_cd"] for c in valid_classes]) if valid_classes else 0.0

    return {
        "model_name": args.config,
        "epoch": selected_epoch,
        "eval_loss": float(eval_loss),
        "eval_coarse_cd": float(eval_coarse_cd),
        "eval_dense_cd": float(eval_dense_cd),
        "macro_avg_loss": float(macro_avg_loss),
        "macro_avg_coarse_cd": float(macro_avg_coarse_cd),
        "macro_avg_dense_cd": float(macro_avg_dense_cd),
        "class_results": class_results
    }

print("Running evaluation...")
results = evaluate(model, test_loader)
print(f"Evaluation finished: Loss={results['eval_loss']:.6f}, Coarse CD={results['eval_coarse_cd']:.6f}, Dense CD={results['eval_dense_cd']:.6f}")

# Save results
output_dir = os.path.join(BASE_DIR, "evaluation", args.run)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "results.json")

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {output_file}")

# Generate test examples
examples_output_dir = os.path.join(output_dir, "test_examples")
os.makedirs(examples_output_dir, exist_ok=True)

# Select random examples for visualization
num_examples = 40
# Sample from the subset or full dataset that we evaluated on
indices = np.random.choice(
    len(test_dataset),
    size=min(num_examples, len(test_dataset)),
    replace=False
)

idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

# Helper to setup 3D subplot
def add_subplot_3d(fig, pos, pts, title, color):
    ax = fig.add_subplot(pos, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=color, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=45)
    return ax

for plot_idx, sample_idx in enumerate(indices):
    partial, complete, label = test_dataset[sample_idx]
    class_name = idx_to_class[label.item()]
    
    inputs = partial.unsqueeze(0).to(device)
    with torch.no_grad():
        coarse_pred, dense_pred = model(inputs)
    
    cd_coarse = chamfer_distance(coarse_pred, complete.unsqueeze(0).to(device)).item()
    cd_dense = chamfer_distance(dense_pred, complete.unsqueeze(0).to(device)).item()
    
    partial_np = partial.numpy()
    coarse_np = coarse_pred.squeeze(0).cpu().numpy()
    dense_np = dense_pred.squeeze(0).cpu().numpy()
    complete_np = complete.numpy()
    
    # Save individual plots (4 subplots side-by-side)
    single_fig = plt.figure(figsize=(16, 4))
    add_subplot_3d(single_fig, 141, partial_np, "Partial Input (1024 pts)", 'red')
    add_subplot_3d(single_fig, 142, coarse_np, f"Coarse Pred (1024 pts)\nCD: {cd_coarse:.5f}", 'orange')
    add_subplot_3d(single_fig, 143, dense_np, f"Dense Pred\nCD: {cd_dense:.5f}", 'blue')
    add_subplot_3d(single_fig, 144, complete_np, "Ground Truth (16384 pts)", 'green')
    
    single_fig.suptitle(f"Example {plot_idx+1} - Class: {class_name}", fontsize=14)
    plt.tight_layout()
    
    filename = os.path.join(examples_output_dir, f"example_{plot_idx+1}.png")
    single_fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(single_fig)

# Save grid comparison (5 examples, 3 columns: Partial, Prediction, GT)
grid_fig = plt.figure(figsize=(12, 18))
grid_indices = indices[:5]

for row_idx, sample_idx in enumerate(grid_indices):
    partial, complete, label = test_dataset[sample_idx]
    class_name = idx_to_class[label.item()]
    
    inputs = partial.unsqueeze(0).to(device)
    with torch.no_grad():
        _, dense_pred = model(inputs)
        
    partial_np = partial.numpy()
    dense_np = dense_pred.squeeze(0).cpu().numpy()
    complete_np = complete.numpy()
    
    # Input
    ax_in = grid_fig.add_subplot(5, 3, row_idx * 3 + 1, projection='3d')
    ax_in.scatter(partial_np[:, 0], partial_np[:, 1], partial_np[:, 2], s=1, c='red', alpha=0.6)
    if row_idx == 0:
        ax_in.set_title("Partial Input", fontsize=12)
    ax_in.set_ylabel(class_name, fontsize=12, labelpad=10)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_zticks([])
    ax_in.view_init(elev=20, azim=45)
    
    # Pred
    ax_pred = grid_fig.add_subplot(5, 3, row_idx * 3 + 2, projection='3d')
    ax_pred.scatter(dense_np[:, 0], dense_np[:, 1], dense_np[:, 2], s=1, c='blue', alpha=0.6)
    if row_idx == 0:
        ax_pred.set_title("Completed Prediction", fontsize=12)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    ax_pred.set_zticks([])
    ax_pred.view_init(elev=20, azim=45)
    
    # GT
    ax_gt = grid_fig.add_subplot(5, 3, row_idx * 3 + 3, projection='3d')
    ax_gt.scatter(complete_np[:, 0], complete_np[:, 1], complete_np[:, 2], s=1, c='green', alpha=0.6)
    if row_idx == 0:
        ax_gt.set_title("Ground Truth", fontsize=12)
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    ax_gt.set_zticks([])
    ax_gt.view_init(elev=20, azim=45)

plt.tight_layout()
grid_filename = os.path.join(examples_output_dir, "completion_examples.png")
grid_fig.savefig(grid_filename, dpi=300, bbox_inches="tight")
plt.close(grid_fig)
print(f"Plots saved to {examples_output_dir}")

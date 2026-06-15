import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from plyfile import PlyData

def load_ply(path):
    v = PlyData.read(path)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

def resample_pcd(pcd, n):
    idx = np.random.permutation(len(pcd))
    if len(idx) < n:
        pad = np.random.randint(len(pcd), size=(n - len(pcd)))
        idx = np.concatenate([idx, pad])
    return pcd[idx[:n]]

def build_index(root_dir):
    samples = []
    for cid in os.listdir(root_dir):
        cpath = os.path.join(root_dir, cid)
        if os.path.isdir(cpath):
            samples.extend([(os.path.join(cpath, f), cid) for f in os.listdir(cpath) if f.endswith(".ply") and "_" not in f])
    return sorted(samples)

class ShapeNetPLYDataset(Dataset):
    def __init__(self, data_dir, split="train", num_input_points=1024, num_gt_points=16384, num_views=8):
        self.complete_dir = os.path.join(data_dir, split, "complete")
        self.partial_dir = os.path.join(data_dir, split, "partial")
        self.n_in, self.n_gt, self.num_views = num_input_points, num_gt_points, num_views
        
        self.samples = build_index(self.complete_dir)
        class_ids = sorted(list(set(c for _, c in self.samples)))
        self.class_to_idx = {c: i for i, c in enumerate(class_ids)}
        self.num_classes = len(class_ids)

    def __len__(self):
        return len(self.samples) * self.num_views

    def __getitem__(self, idx):
        # divmod cleanly handles both the model index and the view ID
        model_idx, view_id = divmod(idx, self.num_views)
        complete_path, class_id = self.samples[model_idx]
        
        base_name = os.path.splitext(os.path.basename(complete_path))[0]
        rel_path = os.path.relpath(complete_path, self.complete_dir)
        partial_path = os.path.join(self.partial_dir, os.path.dirname(rel_path), f"{base_name}_{view_id}.ply")

        partial = resample_pcd(load_ply(partial_path), self.n_in)
        complete = resample_pcd(load_ply(complete_path), self.n_gt)
        label = self.class_to_idx[class_id]

        return torch.from_numpy(partial).float(), torch.from_numpy(complete).float(), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    dataset = ShapeNetPLYDataset(data_dir="data/shapenet", split="train")
    partial, complete, label = dataset[0]

    print(f"Size: {len(dataset)} | Classes: {dataset.num_classes}")
    print(f"Partial: {partial.shape} | Complete: {complete.shape} | Label: {label.item()}")

    # Condensed visualization block
    fig = plt.figure()
    ax1, ax2 = fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")
    ax1.scatter(*partial.T, s=2)
    ax2.scatter(*complete.T, s=2)
    plt.show()
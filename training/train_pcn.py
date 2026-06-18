import os
import sys
import json
import argparse
import socket
import re
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# args
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="config JSON filename")
parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume training from")
args = parser.parse_args()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pcn_provider import ShapeNetPLYDataset
from models.pcn.pcn import PCN, pcn_loss

with open(os.path.join(PROJECT_ROOT, f'configs/{args.config}.json'), 'r') as f:
    config = json.load(f)

SEED = config['seed']
PHYSICAL_BATCH_SIZE = config['physical_batch_size']
NUM_COARSE_POINTS = config['num_coarse_points']
GLOBAL_FEAT_DIM = config['global_feat_dim']
GRID_SIZE = config['grid_size']
GRID_SCALE = config['grid_scale']
MAX_EPOCH = config['max_epoch']
BASE_LEARNING_RATE = config['learning_rate']
GPU_INDEX = config['gpu']
DECAY_STEP = config['decay_step']
DECAY_RATE = config['decay_rate']

TARGET_BATCH_SIZE = 32
ACCUMULATION_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if args.resume:
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint not found at: {args.resume}")
    
    LOG_DIR = os.path.dirname(os.path.abspath(args.resume))
    
    filename = os.path.basename(args.resume)
    match = re.match(r"pcn_epoch_(\d+)\.pth", filename)
    if match:
        start_epoch = int(match.group(1))
    else:
        raise ValueError(f"Could not parse epoch number from checkpoint filename: {filename}. Expected format: pcn_epoch_<number>.pth")
    
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
    LOG_FOUT.write(f"\n--- Resuming training from epoch {start_epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
else:
    start_epoch = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = os.path.join(PROJECT_ROOT, config["log_dir"], timestamp)
    os.makedirs(LOG_DIR, exist_ok=True)

    os.system(f'cp {os.path.abspath(__file__)} {LOG_DIR}') 
    os.system(f'cp {os.path.join(PROJECT_ROOT, "models/pcn/pcn.py")} {LOG_DIR}')

    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(config) + '\n')

def log_string(out_str):
    """Writes to both terminal and physical log file."""
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def train():
    device = torch.device(f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu')
    log_string(f"Training on device: {device} | Hostname: {socket.gethostname()}")
    log_string(f"Physical Batch: {PHYSICAL_BATCH_SIZE} | Accumulation Steps: {ACCUMULATION_STEPS}")

    train_dataset = ShapeNetPLYDataset(data_dir=os.path.join(PROJECT_ROOT, "data/shapenet"), split="train")
    val_dataset = ShapeNetPLYDataset(data_dir=os.path.join(PROJECT_ROOT, "data/shapenet"), split="valid", num_views=1)

    train_loader = DataLoader(train_dataset, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=PHYSICAL_BATCH_SIZE * 2, shuffle=False, num_workers=4)

    model = PCN(num_coarse=NUM_COARSE_POINTS, global_feat_dim=GLOBAL_FEAT_DIM, grid_size=GRID_SIZE, grid_scale=GRID_SCALE).to(device)
    
    if args.resume:
        log_string(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)

    optimizer = optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=DECAY_STEP, gamma=DECAY_RATE)

    train_writer = SummaryWriter(os.path.join(LOG_DIR, 'train'))
    test_writer = SummaryWriter(os.path.join(LOG_DIR, 'test'))

    global_step = 0

    if args.resume:
        log_string(f"Catching up scheduler and global_step for {start_epoch} epochs...")
        for ep in range(start_epoch):
            for batch_idx in range(len(train_loader)):
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    scheduler.step()
                global_step += 1
        log_string(f"Resumed at global_step: {global_step}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    for epoch in range(start_epoch, MAX_EPOCH):
        log_string(f'\n**** EPOCH {epoch+1:03d} ****')
        sys.stdout.flush()

        if epoch < 10: alpha = 0.01
        elif epoch < 20: alpha = 0.1
        elif epoch < 30: alpha = 0.5
        else: alpha = 1.0

        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1:03d}/{MAX_EPOCH:03d}', leave=False)

        for batch_idx, (partial, complete, _) in enumerate(train_pbar):
            partial, complete = partial.to(device), complete.to(device)

            coarse_pred, fine_pred = model(partial)
            loss = pcn_loss(coarse_pred, fine_pred, complete, alpha)

            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            real_loss_value = loss.item() * ACCUMULATION_STEPS
            total_train_loss += real_loss_value
            current_lr = optimizer.param_groups[0]['lr']

            train_writer.add_scalar('loss', real_loss_value, global_step)
            train_writer.add_scalar('learning_rate', current_lr, global_step)
            train_writer.add_scalar('alpha', alpha, global_step)

            train_pbar.set_postfix({'Loss': f'{real_loss_value:.4f}', 'LR': f'{current_lr:.6f}'})

            if batch_idx % 100 == 0:
                LOG_FOUT.write(f"Batch [{batch_idx:05d}/{len(train_loader):05d}] | Loss: {real_loss_value:.4f} | LR: {current_lr:.6f}\n")
                LOG_FOUT.flush()

            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        log_string(f'Mean Train Loss: {avg_train_loss:.6f}')

        model.eval()
        total_val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f'Eval Epoch {epoch+1:03d}/{MAX_EPOCH:03d}', leave=False)
        
        with torch.no_grad():
            for partial, complete, _ in val_pbar:
                partial, complete = partial.to(device), complete.to(device)
                coarse_pred, fine_pred = model(partial)
                loss = pcn_loss(coarse_pred, fine_pred, complete, alpha)
                
                total_val_loss += loss.item()
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        test_writer.add_scalar('eval/loss', avg_val_loss, global_step)
        log_string(f'Eval Mean Loss: {avg_val_loss:.6f}')

        if (epoch + 1) % 1 == 0 or (epoch + 1) == MAX_EPOCH:
            save_path = os.path.join(LOG_DIR, f"pcn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            log_string(f"Model checkpoint saved in: {save_path}")

    train_writer.close()
    test_writer.close()

if __name__ == "__main__":
    try:
        train()
    finally:
        LOG_FOUT.close()
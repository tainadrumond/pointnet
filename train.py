from datetime import datetime
import json
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import socket
import importlib
import os
import sys
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="config JSON filename"
)
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

# Lê os argumentos do arquivo de configuração
with open(f'configs/{args.config}.json', 'r') as f:
    config = json.load(f)

SEED = config['seed']
BATCH_SIZE = config['batch_size']
NUM_POINT = config['num_point']
MAX_EPOCH = config['max_epoch']
BASE_LEARNING_RATE = config['learning_rate']
GPU_INDEX = config['gpu']
MOMENTUM = config['momentum']
OPTIMIZER = config['optimizer']
DECAY_STEP = config['decay_step']
DECAY_RATE = config['decay_rate']

# seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

MODEL_NAME = config['model']
MODEL = importlib.import_module(MODEL_NAME) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', MODEL_NAME + '.py')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(
    config["log_dir"],
    timestamp
)

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(config)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    # Tradução do tf.train.exponential_decay(staircase=True)
    learning_rate = BASE_LEARNING_RATE * (DECAY_RATE ** (batch * BATCH_SIZE // DECAY_STEP))
    return max(learning_rate, 0.00001)

def get_bn_decay(batch):
    bn_momentum = BN_INIT_DECAY * (BN_DECAY_DECAY_RATE ** (batch * BATCH_SIZE // BN_DECAY_DECAY_STEP))
    bn_decay = min(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def apply_bn_decay(model, bn_decay):
    # No PyTorch, o momentum do BatchNorm é análogo a (1 - bn_decay) do TensorFlow
    bn_momentum = 1 - bn_decay
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = bn_momentum

def train():
    device = torch.device(f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu')
    
    # Baseado nas conversões anteriores, o modelo foi empacotado na classe PointNetCls
    classifier = MODEL.PointNetCls(num_classes=NUM_CLASSES).to(device)

    # Optimizer
    if OPTIMIZER == 'momentum':
        optimizer = optim.SGD(classifier.parameters(), lr=BASE_LEARNING_RATE, momentum=MOMENTUM)
    elif OPTIMIZER == 'adam':
        optimizer = optim.Adam(classifier.parameters(), lr=BASE_LEARNING_RATE)

    # Summary writers (TensorBoard)
    train_writer = SummaryWriter(os.path.join(LOG_DIR, 'train'))
    test_writer = SummaryWriter(os.path.join(LOG_DIR, 'test'))

    global_step = 0

    for epoch in range(MAX_EPOCH):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()
         
        global_step = train_one_epoch(classifier, optimizer, device, train_writer, global_step)
        eval_one_epoch(classifier, device, test_writer, global_step)
        
        # Save the variables to disk
        if epoch % 10 == 0:
            save_path = os.path.join(LOG_DIR, "model.pth")
            torch.save(classifier.state_dict(), save_path)
            log_string("Model saved in file: %s" % save_path)


def train_one_epoch(classifier, optimizer, device, train_writer, global_step):
    # is_training = True
    classifier.train()
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            
            inputs = torch.tensor(jittered_data, dtype=torch.float32).to(device)
            labels = torch.tensor(current_label[start_idx:end_idx], dtype=torch.long).to(device)

            # Atualização manual do Learning Rate baseada no seu script original
            lr = get_learning_rate(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Atualização do BN Decay
            bn_decay = get_bn_decay(global_step)
            apply_bn_decay(classifier, bn_decay)

            optimizer.zero_grad()

            pred, end_points = classifier(inputs)
            loss = MODEL.get_loss(pred, labels, end_points)
            
            loss.backward()
            optimizer.step()

            pred_val = pred.argmax(dim=1).cpu().numpy()
            labels_val = labels.cpu().numpy()

            correct = np.sum(pred_val == labels_val)
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss.item()

            train_writer.add_scalar('loss', loss.item(), global_step)
            train_writer.add_scalar('accuracy', correct / float(BATCH_SIZE), global_step)
            train_writer.add_scalar('learning_rate', lr, global_step)
            train_writer.add_scalar('bn_decay', bn_decay, global_step)

            global_step += 1
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

    return global_step


def eval_one_epoch(classifier, device, test_writer, global_step):
    # is_training = False
    classifier.eval()
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    with torch.no_grad():
        for fn in range(len(TEST_FILES)):
            log_string('----' + str(fn) + '-----')
            current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
            current_data = current_data[:, 0:NUM_POINT, :]
            current_label = np.squeeze(current_label)
            
            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE

                inputs = torch.tensor(current_data[start_idx:end_idx, :, :], dtype=torch.float32).to(device)
                labels = torch.tensor(current_label[start_idx:end_idx], dtype=torch.long).to(device)

                pred, end_points = classifier(inputs)
                loss = MODEL.get_loss(pred, labels, end_points)

                pred_val = pred.argmax(dim=1).cpu().numpy()
                labels_val = labels.cpu().numpy()

                correct = np.sum(pred_val == labels_val)
                total_correct += correct
                total_seen += BATCH_SIZE
                loss_sum += (loss.item() * BATCH_SIZE)
                
                for i in range(start_idx, end_idx):
                    l = current_label[i]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    eval_loss = loss_sum / float(total_seen)
    eval_acc = total_correct / float(total_seen)
    # np.float foi removido em versões mais novas do NumPy, substituído pelo 'float' nativo do Python para evitar quebras.
    eval_class_acc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=float))

    test_writer.add_scalar('eval/loss', eval_loss, global_step)
    test_writer.add_scalar('eval/accuracy', eval_acc, global_step)

    log_string('eval mean loss: %f' % eval_loss)
    log_string('eval accuracy: %f' % eval_acc)
    log_string('eval avg class acc: %f' % eval_class_acc)


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
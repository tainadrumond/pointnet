import os
import sys
import importlib
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="config JSON filename"
)
parser.add_argument(
    "--run",
    type=str,
    required=True,
    help="run name"
)
args = parser.parse_args()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import scripts.pointnet_provider as provider

# load settings
with open(f'configs/{args.config}.json', 'r') as f:
    config = json.load(f)

# read settings
NUM_CLASSES=40
SEED = config['seed']
BATCH_SIZE = config['batch_size']
NUM_POINT = config['num_point']
MODEL_NAME = config['model']

# seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# load test dataset
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

# load model
MODEL = importlib.import_module(f"pointnet.{MODEL_NAME}") # import network module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MODEL.PointNetCls(num_classes=NUM_CLASSES)
checkpoint = torch.load(f"log/{args.run}/model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)

# evaluate
def evaluate(model):
    model.eval()

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    with torch.no_grad():
        for fn in range(len(TEST_FILES)):
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

                pred, end_points = model(inputs)
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
    eval_class_acc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
    #print('eval mean loss: %f' % eval_loss)
    #print('eval accuracy: %f' % eval_acc)
    #print('eval avg class acc: %f' % eval_class_acc)

    return {
        "model_name": args.config,
        "eval_loss": float(eval_loss),
        "eval_accuracy": float(eval_acc),
        "eval_avg_class_accuracy": float(eval_class_acc)
    }

results = evaluate(model)


output_dir = os.path.join(
    BASE_DIR,
    "evaluation",
    args.run
)
os.makedirs(output_dir, exist_ok=True)

# save metrics
output_file = os.path.join(
    output_dir,
    "results.json"
)

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

# test examples

examples_output_dir = os.path.join(
    BASE_DIR,
    "evaluation",
    args.run,
    "test_examples"
)
os.makedirs(examples_output_dir, exist_ok=True)

with open(
    os.path.join(BASE_DIR, "data/modelnet40_ply_hdf5_2048/shape_names.txt")
) as f:
    class_names = [line.strip() for line in f]

current_data, current_label = provider.loadDataFile(TEST_FILES[0])

current_label = np.squeeze(current_label)

num_examples = 10

indices = np.random.choice(
    len(current_data),
    size=num_examples,
    replace=False
)

fig = plt.figure(figsize=(20, 8))

for plot_idx, sample_idx in enumerate(indices):

    points = current_data[sample_idx]
    true_label = int(current_label[sample_idx])

    inputs = torch.tensor(
        points[np.newaxis, :, :],
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        pred, _ = model(inputs)

    pred_label = pred.argmax(dim=1).item()

    correct = pred_label == true_label

    title = (
        f"GT: {class_names[true_label]}\n"
        f"Pred: {class_names[pred_label]}\n"
        f"{'(V)' if correct else '(X)'}"
    )

    single_fig = plt.figure(figsize=(5, 5))
    single_ax = single_fig.add_subplot(
        111,
        projection="3d"
    )

    single_ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=2
    )

    single_ax.set_title(title)

    single_ax.set_xticks([])
    single_ax.set_yticks([])
    single_ax.set_zticks([])

    single_ax.view_init(
        elev=0,
        azim=0
    )

    filename = os.path.join(
        examples_output_dir,
        f"example_{plot_idx+1}.png"
    )

    single_fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(single_fig)

    ax = fig.add_subplot(
        2,
        5,
        plot_idx + 1,
        projection="3d"
    )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=2
    )

    ax.set_title(
        title,
        fontsize=9
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(
        elev=0,
        azim=0
    )

# salvar grade completa
grid_filename = os.path.join(
    examples_output_dir,
    "classification_examples.png"
)

plt.tight_layout()

fig.savefig(
    grid_filename,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)
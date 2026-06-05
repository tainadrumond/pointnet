import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetCls(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNetCls, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        
        # tf 0.7 equivale a p=0.3 no dropout do PyTorch
        self.dropout = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # (B, N, 3) --> (B, 3, N) para Conv1d
        x = x.transpose(1, 2)
        
        # Point functions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Max pooling na dimensão dos pontos para obter a feature global (B, 1024, 1)
        x = torch.max(x, 2, keepdim=True)[0]
        
        # achatar para (B, 1024) para as camadas fully connected
        x = x.view(-1, 1024)
        
        # MLP global
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        end_points = {}
        return x, end_points


def get_loss(pred, label, end_points):
    """ 
    pred: (B, NUM_CLASSES)
    label: (B)
    """
    criterion = nn.CrossEntropyLoss()
    classify_loss = criterion(pred, label)
    return classify_loss


if __name__ == '__main__':
    model = PointNetCls(num_classes=40)
    
    model.train()
    
    # (B, N, 3)
    inputs = torch.zeros((32, 1024, 3))
    
    outputs, _ = model(inputs)
    
    print("Shape do output:", outputs.shape) # (B, num_classes) --> (32, 40)
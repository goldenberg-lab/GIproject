import torch
from torch import nn
import torch.nn.functional as F

# -------------------------------------------------- #
# ------- BINARY CNN FOR MULTITASK OUTCOME --------- #

class CNN_binary(nn.Module):
    def __init__(self,n_tasks):
        super(CNN_binary, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=20,stride=5)
        self.pool1 = nn.MaxPool2d(kernel_size=10, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=10, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2)
        self.fc1 = nn.Linear(in_features= 12 * 16 * 16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=3)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --------------------------------------------------- #
# ------- ORDINAL CNN FOR MULTITASK OUTCOME --------- #

class CNN_ordinal(nn.Module):
    def __init__(self):
        super(CNN_ordinal, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=20,stride=5)
        self.pool1 = nn.MaxPool2d(kernel_size=10, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=10, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2)
        self.fc1 = nn.Linear(in_features= 12 * 16 * 16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=10)
        self.fc3B = nn.Linear(in_features=10, out_features=5)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x =  self.fc3B(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.fc = nn.Linear(1152, 10)

    def forward(self, X):
        Y = self.conv1(X)
        Y = F.max_pool2d(Y, (2, 2))
        Y = F.relu(Y)
        Y = self.conv2(Y)
        Y = F.max_pool2d(Y, (2, 2))
        Y = F.relu(Y)
        Y = self.conv3(Y)
        Y = F.max_pool2d(Y, (2, 2))
        Y = Y.view(Y.size(0), -1)
        Y = self.fc(Y)
        return Y

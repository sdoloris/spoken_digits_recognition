import torch
import torch.nn as nn
import torch.nn.functional as F

# Creating the neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 512, 16)
        self.pool = nn.MaxPool2d(8)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 11)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        output = self(x)
        return torch.max(output,1)[1].squeeze(0).item()
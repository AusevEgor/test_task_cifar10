import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, batch_size, hidden):
        super(Model, self).__init__()
        self.bs = batch_size
        self.fc = nn.Linear(32*32*3, hidden)
        # self.fc1 = nn.Linear(1000, 512)

        # self._fc1 = nn.Linear(512, 1000)
        self._fc = nn.Linear(hidden, 32*32*3)

    def forward(self, x):

        x = x.view(-1, 3*32*32)
        e1 = F.relu(self.fc(x))
        # e2 = F.relu(self.fc1(e1))

        # d1 = F.relu(self._fc1(e2))
        out = F.sigmoid(self._fc(e1))

        out = out.view(-1, 3, 32, 32)

        return out

    def encoder(self, x):
        x = x.view(-1, 3 * 32 * 32)
        e1 = F.relu(self.fc(x))
        # e2 = F.relu(self.fc1(e1))

        return e1

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, hidden):
        super(Conv, self).__init__()
        # self.bs = batch_size

        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 16, 3, padding=1)
        self.mp = nn.MaxPool2d(2)

        self.fc = nn.Linear(16 * 8 * 8, hidden)
        self.cf = nn.Linear(hidden, 16 * 8* 8)

        self.tc1 = nn.ConvTranspose2d(16, 64, 2, stride=2)
        self.tc2 = nn.ConvTranspose2d(64, 3, 2, stride=2)
        # self.fc1 = nn.Linear(1000, 512)

        # self._fc1 = nn.Linear(512, 1000)
        # self._fc = nn.Linear(hidden, 32*32*3)

    def forward(self, x):

        # x = x.view(-1, 3*32*32)
        e1 = F.relu(self.c1(x))
        e1 = self.mp(e1)
        e2 = F.relu(self.c2(e1))
        e2 = self.mp(e2)

        e2 = e2.view(-1, 16* 8 *8)

        hidden = F.relu(self.fc(e2))

        img = F.relu(self.cf(hidden))
        img = img.view(-1, 16, 8, 8)

        d1 = F.relu(self.tc1(img))
        out = F.tanh(self.tc2(d1))

        # d1 = F.relu(self._fc1(e2))
        # out = F.sigmoid(self._fc(e1))

        # out = out.view(-1, 3, 32, 32)

        return out

    def encoder(self, x):
        # x = x.view(-1, 3 * 32 * 32)
        e1 = F.relu(self.c1(x))
        e1 = self.mp(e1)
        e2 = F.relu(self.c2(e1))
        e2 = self.mp(e2)

        e2 = e2.view(-1, 16 * 8 * 8)

        hidden = self.fc(e2)
        # e2 = F.relu(self.fc1(e1))

        return hidden

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, hidden, out):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden, out)
        # self.fc1 = nn.Linear(1000, 512)

        # self._fc1 = nn.Linear(512, 1000)
        # self._fc = nn.Linear(hidden, 32*32*3)

    def forward(self, x):

        # x = x.view(-1, 3*32*32)
        out = self.fc(x)
        # e2 = F.relu(self.fc1(e1))

        # d1 = F.relu(self._fc1(e2))
        # out = F.sigmoid(self._fc(e1))

        # out = out.view(-1, 3, 32, 32)

        return F.log_softmax(out, dim=1)


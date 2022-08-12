import numpy as np
import matplotlib
from torchsummary import summary
from data.data_load import *
from models.UNet import Unet
from models.FL import Model
from models.Conv import *
from models.detector import *
from utils.learning import *

dataset = 'cifar10'
batch_size = 64
lr = 0.001

if dataset == 'cifar10':
    train_loader, test_loader = download_CIFAR10(batch_size=batch_size)
elif dataset == 'cifar100':
    train_loader, test_loader = download_CIFAR100(batch_size=batch_size)
else:
    raise Exception("Dataset not found")

for h in [16, 32, 64, 128, 256, 512]:
    path_model = f'./outputs/conv/{h}/199/model.pth'
    model = Conv(h).to('cuda')
    model.load_state_dict(torch.load(path_model))
    model.eval()
    # summary(model, (3,32,32))
    decoder = Decoder(h, len(test_loader.dataset.classes)).to('cuda')

    # for batch in train_loader:
    #fit(model,200, train_loader, lr, h)
    print('Learning hidden ', h,)
    fit_detector(model, decoder, 100, train_loader, test_loader, lr, 1, h)


from torch.optim import SGD, Adam
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss, NLLLoss
from tqdm import tqdm
from .image import *
import os
import torch
from torchmetrics import Accuracy


def fit(model, epoches, dataloader_train, lr, h):

    opt = Adam(model.parameters(), lr=lr)
    loss = MSELoss()
    path = './outputs/conv/' + str(h) + '/'
    error_list = []
    for e in range(epoches):
        tmp = []
        for i, (x, _) in tqdm(enumerate(dataloader_train)):

            x= x.detach().to('cuda')
            opt.zero_grad()
            out = model(x)

            error = loss(out, x)
            error.backward()
            opt.step()
            tmp.append(error.to('cpu').detach().float())
            if i %100 == 0:
                print(error)
        error_list.append(np.mean(tmp))
        os.makedirs(path + str(e), exist_ok=True)
        torch.save(model.state_dict(), path + str(e) + '/model.pth')
        show(x[0].to('cpu'), e, path + str(e) + '/orig.png')
        show(out[0].cpu().detach(), e, path + str(e) + '/out.png')
    np.save(path + 'loss', np.array(error_list))

def fit_detector(model, detecotr, epoches, dataloader_train, dataloader_test, lr, it, h):
    opt = Adam(detecotr.parameters(), lr=lr)
    loss = NLLLoss()
    model.eval()
    metric = Accuracy().to('cuda')
    loss_array = []
    train_acc = []
    for e in range(epoches):
        detecotr.train()
        l = []
        aaa = []
        for i, (x, y) in enumerate(dataloader_train):
            # y_ = F.one_hot(y, 10).to('cuda')
            y = y.to('cuda')
            opt.zero_grad()
            x= x.detach().to('cuda')
            x = model.encoder(x)
            # out = model(x)

            out = detecotr(x)
            error = loss(out, y)
            error.backward()
            opt.step()
            l.append(error.to('cpu').detach().numpy())
            aaa.append(float(metric(torch.argmax(out, dim = 1), y)))
            # break
        loss_array.append(np.mean(l))
        train_acc.append(np.mean(aaa))
        print(f'Mean Error {e} epoch = {np.mean(l)}')
        if e % it == 0:
            detecotr.eval()
            acc = []
            logs = []
            yy = []
            for x, y in dataloader_test:
                y = y.to('cuda')
                x = x.detach().to('cuda')
                x = model.encoder(x)
                out = detecotr(x)
                if logs == []:
                    logs = out.to('cpu').detach().float()
                else:
                    logs = np.concatenate([logs, out.to('cpu').detach().float()], axis = 0)
                if yy == []:
                    yy = y.to('cpu').detach().float()
                else:
                    yy = np.concatenate([yy, y.to('cpu').detach().float()], axis = 0)
                out = torch.argmax(out, dim = 1)
                acc.append(float(metric(out, y)))
            print('Test_accuracy = ', np.mean(acc))
        os.makedirs(f'./outputs/detector/{h}/{e}', exist_ok=True)
        np.save(f'./outputs/detector/{h}/{e}/loss', loss_array)
        np.save(f'./outputs/detector/{h}/{e}/train_acc', train_acc)
        np.save(f'./outputs/detector/{h}/{e}/logits', logs)
        np.save(f'./outputs/detector/{h}/{e}/orig_y', yy)
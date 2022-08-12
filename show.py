import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score

path = './outputs/detector/'

hiddens = ['16', '32', '64', '128', '256', '512']
classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
logits = dict()
loss = dict()
orig_y = dict()
train_acc = dict()
for hidden in hiddens:
    logits[hidden] = []
    loss[hidden] = []
    orig_y[hidden] = []
    train_acc[hidden] = []
    for e in os.listdir(path + hidden):
        logits[hidden].append(np.load(os.path.join(path, hidden, e, 'logits.npy')))
        orig_y[hidden].append(np.load(os.path.join(path, hidden, e, 'orig_y.npy')))

        # if e == '99':
        #     loss[hidden].append(np.load(os.path.join(path, hidden, e, 'loss.npy')))
        #
        #     train_acc[hidden].append(np.load(os.path.join(path, hidden, e, 'train_acc.npy')))

print()

pred_y = dict()
for h in hiddens:
    pred_y[h] = np.argmax(logits[h][99], axis=1)
    # for e in logits[h]:
    #     pred_y[h].append(np.argmax(e, axis=1))


matrix = dict()
for h in hiddens:
    matrix[h] = np.zeros((10, 10))
    for i in range(len(pred_y[h])):
        matrix[h][int(orig_y[h][99][i])][pred_y[h][i]] += 1


pression = dict()

for h in hiddens:
    with open(f'./outputs/detector/matrix/confusion_matrix_{h}.csv', 'w') as wr:
        # wr.write(',')
        for cl in classes:
            wr.write(',' + cl)
        wr.write('\n')

        for i in range(len(classes)):
            wr.write(classes[i] + ',')
            for j in range(len(classes)):
                wr.write(str(int(matrix[h][i][j])) + ',')
            wr.write('\n')
# for h in hiddens:
#     pression[h] = dict()
#     for i in range(len(pred_y[h])):
#         if orig_y[h][99][i] not in pression[h].keys():
#             pression[h][orig_y[h][99][i]] = [0,0]
#         if pred_y[h][i] == orig_y[h][99][i]:
#             pression[h][orig_y[h][99][i]][0] += 1
#         pression[h][orig_y[h][99][i]][1] += 1
#
# for h in hiddens:
#     for k, v in pression[h].items():
#         pression[h][k] = int(v[0]/ v[1] * 100)

# test_acc = dict()
# for h in hiddens:
#     test_acc[h] = []
#     for e in range(100):
#         test_acc[h].append(accuracy_score(pred_y[h][e], orig_y[h][e]))
# ar = []
# for h in hiddens:
#     tmp = []
#     for i in range(10):
#         tmp.append(pression[h][i])
#     ar.append(tmp)
#
#
#
# print()
#
#
# fig, ax = plt.subplots()
#
# z = np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.])
#
# for h in range(6):
#     ax.bar(z + (h * 0.1 ), ar[h], width = 0.1)
#     # print(pression[h])
#
# # plt.bar(classes, ar)
# #     plt.plot(loss[hidden])
#
# plt.legend(hiddens, loc = 4)
#
#
# ax.set_xticks(np.arange(len(classes)))
# ax.set_xticklabels(classes)
# plt.ylabel('Precision')
# plt.grid()
# plt.title('Test detrctor precision for different hiddens sizes')
# plt.savefig(path + 'Test_precision.png')
# plt.show()
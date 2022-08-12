import matplotlib.pyplot as plt
import numpy as np

def show(img, e, name):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(name + '.png')
    # plt.show()

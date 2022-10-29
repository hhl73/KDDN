import numpy as np
import argparse
from dataset.data import dehazeDataloader
from network.Student import *


modelG = Student2()
#数据集
dataset = dehazeDataloader(train=True, transform=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

def normImage(image, num=1.):
    if len(image.shape) > 2:
        for i in range(3):
            img = image[:, :, i]
            max = np.max(img)
            min = np.min(img)
            image[:, :, i] = (img - min) / (max - min + 1e-8)
    else:
        max = np.max(image)
        min = np.min(image)
        image = (image - min) / (max - min + 1e-8) * num
    return image

modelG = torch.load(".\\models\\Student\\Student.pkl")
import matplotlib.pyplot as plt

for (testin, testlabel) in dataloader:
    testout, _ = modelG(testin.cuda())
    testin = testin.cpu().numpy().reshape((3, 256, 256)).transpose([1, 2, 0])
    testout = testout.detach().cpu().numpy().reshape((3,256,256)).transpose([1, 2, 0])
    testlabel = testlabel.cpu().numpy().reshape((3, 256, 256)).transpose([1, 2, 0])
    testout = normImage(testout)
    plt.subplot(131)
    plt.imshow(testin)
    plt.subplot(132)
    plt.imshow(testout)
    plt.subplot(133)
    plt.imshow(testlabel)
    plt.show()
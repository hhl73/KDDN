import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import re
import random


def make_dataset():#获得shuffle过的hazy和clear图片的路径
    hazyImages = []
    clearImages = []
    dataset = "\\dataset\\archive\\"
    abpath = os.getcwd()
    path = abpath+dataset #数据的路径
    for i in range(1, 1399):
        clearImages.append(path + "clear\\" + str(i) + ".png")
        #hazyImages.append(dataset+"trains/"+str(i)+"_1.png")
        #从有雾的图像里找一个配对
        fl = os.listdir(path + "hazy\\")
        for item in fl:
            if re.match(str(i) + "_10_", item):
                hazyImages.append(path + "hazy\\" + item)
                break
    #打乱
    indices = np.arange(len(clearImages))  # np.arange(99)#
    np.random.shuffle(indices)
    clearShuffle = []
    hazyShuffle = []
    for i in range(len(indices)):
        index = indices[i]
        clearShuffle.append(clearImages[index])
        hazyShuffle.append(hazyImages[index])

    return clearShuffle, hazyShuffle

#对传入的图片list进行随机旋转
#传入images的list
def random_rot(images):
    randint = random.randint(0, 4)
    if randint == 0:
        for i in range(len(images)):
            images[i] = cv2.rotate(images[i], cv2.ROTATE_90_CLOCKWISE)
    elif randint == 1:
        for i in range(len(images)):
            images[i] = cv2.rotate(images[i], cv2.ROTATE_180)
    elif randint == 2:
        for i in range(len(images)):
            images[i] = cv2.rotate(images[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        pass
    return images

#随机裁剪到256
#传入image的list
def random_crop(images, sizeTo=256):
    w = images[0].shape[1]
    h = images[0].shape[0]
    w_offset = random.randint(0, max(0, w - sizeTo - 1))
    h_offset = random.randint(0, max(0, h - sizeTo - 1))

    for i in range(len(images)):
        images[i] = images[i][h_offset:h_offset + sizeTo, w_offset:w_offset + sizeTo, :]
    return images

#随机翻转
#传入image的list
def random_flip(images):
    if random.random() < 0.5:
        for i in range(len(images)):
            images[i] = cv2.flip(images[i], 1)
    if random.random() < 0.5:
        for i in range(len(images)):
            images[i] = cv2.flip(images[i], 0)
    return images

#传入image的list
def image_resize(images, siezeTo=(256,256)):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], siezeTo)
    return images

#标准化image
def normImge(image, num=1.):
    if len(image.shape) > 2:
        for i in range(3):
            img = image[:,:,i]
            max = np.max(img)
            min = np.min(img)
            image[:, :, i] = (img - min)/(max - min + 1e-8)
    else:
        max = np.max(image)
        min = np.min(image)
        image = (image - min) / (max - min + 1e-8) * num
    return image

class dehazeDataloader(Dataset):
    def __init__(self, train=True, transform=None):
        clearImages, hazyImages = make_dataset()
        self.images = hazyImages
        self.clearImages = clearImages
        self._transform = transform

    def __getitem__(self, index):
        Ix = Image.open(self.images[index]).convert('RGB')
        Ix = np.array(Ix, dtype=np.float64) / 255.

        Jx = Image.open(self.clearImages[index]).convert('RGB')
        Jx = np.array(Jx, dtype=np.float64) / 255.

        images = [Ix, Jx]

        images = random_crop(images, 256)
        # images = image_resize(images, (256, 256))

        images = random_rot(images)
        images = random_flip(images)

        [Ix, Jx] = images

        if self._transform is not None:
            Ix, Jx = self.transform(Ix, Jx)

        return Ix, Jx

    def __len__(self):
        return len(self.images)

    def transform(self, Ix, Jx):
        #plt.imshow(img, cmap=plt.cm.gray), plt.show()
        Ix = Ix.transpose([2, 0, 1])#改变RGB序列
        Ix = torch.from_numpy(Ix).float()

        Jx = Jx.transpose([2, 0, 1])
        Jx = torch.from_numpy(Jx).float()

        return Ix, Jx


class myDataloader():
    def __init__(self):
        trainset = dehazeDataloader(train=True, transform=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=8)

        self.trainloader = trainloader

    def getLoader(self):
        return self.trainloader

if __name__ =="__main__":

    trainLoader = dehazeDataloader(train=True, transform=True)

    for index, (Ix, Jx) in enumerate(trainLoader):

        Ix = Ix.numpy()
        Ix = Ix.transpose([1, 2, 0])

        Jx = Jx.numpy()
        Jx = Jx.transpose([1, 2, 0])

        plt.subplot(221), plt.imshow(Ix)
        plt.subplot(222), plt.imshow(Jx)
        plt.show()
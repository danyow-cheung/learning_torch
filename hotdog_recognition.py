import os 
import torch
import torchvision
from torch import nn 
from d2l import torch as d2l 
import matplotlib.pyplot as plt 
# 获取数据集
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip','fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_iter = d2l.download_extract('hotdog')
# 创建两个实例来分别读取训练和测试中的图像文件
train_images = torchvision.datasets.ImageFolder(os.path.join(data_iter,'train'))
test_images = torchvision.datasets.ImageFolder(os.path.join(data_iter,'test'))
hotdogs = [train_images[i][0] for i in range(8)]
not_hotdogs = [train_images[-i-1][0] for i in range(8)]
# plt.subplot(1,2,1)
# plt.plot(hotdogs)
# plt.subplot(1,2,2)
# plt.plot(not_hotdogs)
# plt.show()
# d2l.show_images(hotdogs+not_hotdogs,2,8,scale=1.4)
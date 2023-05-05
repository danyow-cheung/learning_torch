import torch
import  torchvision
from torch import nn 
from torch.nn import functional as F 
from d2l import torch as d2l 

# 下 面， 我 们 使 用 在ImageNet数 据 集 上 预 训 练 的ResNet-18模 型 来 提 取 图 像 特 征， 并 将 该 网 络 记 为pretrained_net。ResNet-18模型的最后几层包括全局平均汇聚层和全连接层，然而全卷积网络中不 需要它们。

pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
# 创建全卷积网络net，复制了ResNet-18大部分预训练层
# 除了最后的全局平均汇聚和最接近输出的全连接层
net = nn.Sequential(*list(pretrained_net.children())[:-2])
# 给定1
# 高度为320，宽度480的shu r
X  = torch.rand(size=(1,3,320,480))
print(net(X).shape) 
num_classes = 21 
net.add_module('final_conv',nn.Conv2d(512,num_classes,kernel_size=1))
net.add_module('transpose_conv',nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,padding=16,stride=32))

'''初始化转追卷积层'''
# 双线性插值
def bilinear_kernel(in_channels,out_channels,kernel_size):
    factor = (kernel_size+1)//2 
    if kernel_size%2==1:
        center = factor -1 
    else:
        center = factor-0.5 
    og = (torch.arange(kernel_size).reshape(-1,1),torch.arange(kernel_size).reshape(1,-1))
    filt = (1-torch.abs(og[0]-center)/factor)*(1-torch.abs(og[1]-center)/factor)

    weight = torch.zeros((in_channels,out_channels,kernel_size,kernel_size))
    weight[range(in_channels),range(out_channels),:,:,] = filt 
    return weight


if __name__ =='__main__':
    batch_size = 32 
    crop_size = (320,480)
    train_iter ,test_iter = d2l.load_data_voc(batch_size,crop_size)
    def loss(inputs,targets):
        return F.cross_entropy(inputs,targets,reduction='none').mean().mean(1)
    
    num_epochs = 1 
    lr = 0.001 
    wd = 1e-3 
    device = d2l.try_gpu()

    trainer = torch.optim.SGD(net.parameters(),lr=lr,weight_decay=wd)
    d2l.train_ch3(net,train_iter,test_iter,loss,trainer,num_epochs,device)

    def predict(img):
        X = test_iter.dataset.normalize_image(img).unsqueeze(0)
        pred = net(X.to(device[0])).argmax(dim=1)
        return pred.reshape(pred.shape[1],pred.shape[2])
    
    def label2image(pred):
        '''预测类别映射回它们在数据集中标注颜色'''
        colormap = torch.tensor(d2l.VOC_COLORMAP,device=device)
        X = pred.long()
        return colormap[X,:]
    
    # 测试数据集中的图像大小，形状各异，由于模型使用了步幅为32的转置卷积层，因此当输入图像
    # 高或宽无法被32整除，转置卷积输出的高或宽会与输入图像的尺寸有偏差
    voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

    test_images ,test_labels = d2l.read_voc_images(voc_dir,False)
    n = 4 
    imgs = []
    for i in range(n):
        crop_rect = (0,0,320,480)
        X = torchvision.transforms.functional.crop(test_images[i],*crop_rect)
        pred = label2image(predict(X))
        imgs += [X.permute(1,2,0),pred.cpu(),torchvision.transforms.functional.crop(
            test_labels[i],*crop_rect
        ).permute(1,2,0)]
    
    d2l.show_images(imgs[:3]+imgs[1::3]+imgs[2::3],3,n,scale=2)
    
import os 
import torch 
import torchvision
from torch import nn 
import pandas as pd 
from d2l import torch as d2l

'''导入数据集'''
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL+'kaggle_dog_tiny.zip',
                                '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d'
                                )

# 使用小批量
data_dir = d2l.download_extract('dog_tiny')

def reorg_dog_data(data_dir,valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir,'labels.csv'))
    d2l.reorg_train_valid(data_dir,labels,valid_ratio)
    d2l.reorg_test(data_dir)

batch_size = 32 
valid_ratio = 0.1 
reorg_dog_data(data_dir,valid_ratio)


'''图像增广'''
transform_train = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08～1.0之间，高宽比在3/4和4/3之间
    torchvision.transforms.RandomResizedCrop(224,scale=(0.08,1.0),
                                             ratio=(3.0/4.0,  4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    torchvision.transforms.ColorJitter(brightness=0.4,contrast=.4,saturation=.4),
    # 随机添加噪声
    torchvision.transforms.ToTensor(),
    # 标准化图像每个通道
    torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 测试，使用确定性的图像预处理操作
transform_test = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08～1.0之间，高宽比在3/4和4/3之间
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    # 标准化图像每个通道
    torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds ,train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir,'train_valid_test',folder),
    transform=transform_train
)for folder in ['train','train_valid']]
valid_ds,test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir,'train_valid_test',folder),
    transform=transform_test
) for folder in ['valid','test']]


train_iter,train_valid_iter = [torch.utils.data.DataLoader(
    dataset,batch_size,shuffle=True,drop_last=True
)for dataset in (train_ds,train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(
    valid_ds,batch_size,shuffle=False,drop_last=True,
)


test_iter = torch.utils.data.DataLoader(
    test_ds,batch_size,shuffle=False,drop_last=False,
)

'''微调预训练模型
这里选择了冻结参数
'''
def get_net(device):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)

    # 定义一个新的输出网络,共有120个输出类别
    finetune_net.output_new = nn.Sequential(
        nn.Linear(1000,256),
        nn.ReLU(),
        nn.Linear(256,120),
        )

    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(device)
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False 
    return finetune_net 

loss = nn.CrossEntropyLoss(reduction='none')
def evaluate_loss(data_iter,net,device):
    l_sum = 0 
    n = 0 
    for features,labels in data_dir:
        features,labels = features.to(device),labels.to(device)
        output = net(features)
        l = loss(output,labels)
        l_sum += l.sum()
        n+= labels.numel()
    return (l_sum/n).to('cpu')


# def train(net,train_)

def train(net,train_iter,valid_iter,num_epochs,lr,wd,devices,lr_period,lr_decay):
    # 只定义小型自定义输出网络
    net = nn.DataParallel(net,device_ids = devices).to(devices[0])
    
    
    trainer = torch.optim.SGD(
        (param for param in net.parameters() if param.requires_grad),
        lr=lr,momentum=0.9,weight_decay=wd)
    
    # 学习率调整器
    scheduler = torch.optim.lr_scheduler.StepLR(trainer,lr_period,lr_decay)
    
    num_batches,timer = len(train_iter),d2l.Timer()
    
    legend = ['train_loss']

    if valid_iter is not None:
        legend.append('valid_loss')

    # 注解
    animator = d2l.Animator(xlabel='epoch',xlim=[1,num_batches],legend=legend)

    for epoch in range(num_epochs):
        metrics = d2l.Accumulator(2)

        for i ,(features,labels) in enumerate(train_iter):
            timer.start()
            features,labels = features.to(devices[0]),labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l  = loss(output,labels).sum()
            l.backward()
            trainer.step()
            metrics.add(l,labels.shape[0])
            timer.stop()
            if (i+1)%(num_batches//5)==0 or i == num_batches-1:
                animator.add(epoch+(i+1)/num_batches,(metrics[0]/metrics[1],None))
        measures = f'train loss {metrics[0]/metrics[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter,net,devices)
            animator.add(epoch+1,(None,valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    
    print(measures + f'\n{metrics[1] * num_epochs / timer.sum():.1f}'f' examples/sec on {str(devices)}')


devices = d2l.try_gpu()
num_epochs = 10 
lr  = 1e-4 
wd = 1e-4 

lr_period ,lr_decay ,net = 2,0.9,get_net(devices)
train(net,train_iter,valid_iter,num_epochs,lr,wd,devices,lr_period,lr_decay)

preds = []
for data,label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])),dim=1)
    preds.extend(output.cpu().detach().numpy())

idx = sorted(os.listdir(os.path.join(data_dir,'train_valid_test','test','unknown')))
with open('submission.csv','w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i ,output in zip(idx,preds):
        f.write(i.split('.')[0] + ',' + ','.join([str(num) for num in output])+'\n')
        
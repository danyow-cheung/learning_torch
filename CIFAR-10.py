import collections
import math 
import os 
import shutil
import pandas as pd 
import torch
import torchvision
from d2l import nn 
from d2l import torch as d2l 
'''kaggle-图像分类比赛 使用ResNet18'''

'''导入数据集'''
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL+'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')


# 使用完整数据
demo = False 
data_dir = '../data/cifar-10'

'''整理数据集
读取csv文件中的标签，将字典不带扩展名的部分映射到标签
'''
def read_csv_labels(fname):
    '''读取fname来给标签字典返回一个文件名'''
    with open(fname,'r') as f:
        # 跳过文件头行
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(",") for l in lines]
    return dict(((name,label) for name,label in tokens))


labels = read_csv_labels(os.path.join(data_dir,'trainLabels.csv'))
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))

'''将验证集从原始训练集中拆分出来'''
def copyfile(filename,target_dir):
    '''将文件复制到目标目录'''
    os.makedirs(target_dir,exist_ok=True)
    shutil.copy(filename,target_dir)

def reorganize_train_valid(data_dir,labels,valid_ratio):
    '''将验证集从原始数据集中拆分出来'''
    # 训练集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1,math.floor(n*valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir,'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir,'train',train_file)
        copyfile(fname,os.path.join(data_dir,'train_valid_test','train_valid',label))
        if label not in label_count or label_count[label]<n_valid_per_label:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','valid',label))

            label_count[label] = label_count.get(label,0) + 1 
        else:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','train',label))
    return n_valid_per_label



def reorganize_test(data_dir):
    '''预测期间整理测试集，方便读取'''
    for test_file in os.listdir(os.path.join(data_dir,'test')):
        copyfile(os.path.join(data_dir,'test',test_file),
                 os.path.join(data_dir,'train_valid_test','test','unknown'))
        

def  reorganize_cifar10_data(data_dir,valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir,'trainLebls.csv'))
    reorganize_train_valid(data_dir,labels,valid_ratio)
    reorganize_test(data_dir)


# 定义超参数
batch_size = 32 
valid_ratio = 0.1 
reorganize_cifar10_data(data_dir,valid_ratio)


# 图像增广
transform_train = torchvision.transforms.Compose([
     #高度和宽度上讲图像放大到40像素的正方形
     torchvision.transforms.Resize(40),
     # 随机裁剪出一个高度和宽度均为40像素的正方形图像
     # 生成一个面积为原始图像面积为0.64～1倍的小正方形
     # 然后将其缩放为高度和宽度均为32像素的正方形
     torchvision.transforms.RandomResizedCrop(
        32,scale=(0.64,1.0),ratio=(1.0,1.0)
     ),
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor(),
     # 标准化图像的每一个通道
     torchvision.transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])

])

# 测试期间，只对图像执行标准化，以消除评估结果中的随机性
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])
])


'''读取数据集'''
train_ds,train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir,'train_valid_test',folder),
    transform=transform_train) for folder in ['train','train_valid']]

valid_ds,test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir,'train_valid_test',folder),
    transform=transform_test) for folder in ['valid','test']]
'''
在训练期间，我们需要指定上面定义的所有图像增广操作。
当验证集在超参数调整过程中用于模型评估时， 不应引入图像增广的随机性。在
最终预测之前，我们根据训练集和验证集组合而成的训练模型进行训练，以 充分利用所有标记的数据。
'''
train_iter,train_valid_iter = [torch.utils.data.DataLoader(
    dataset,batch_size,shuffle=True,drop_last=True
    ) for dataset in (train_ds,train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds,batch_size,shuffle=False,drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds,batch_size,shuffle=False,drop_last=True)



"""定义模型
"""
def get_net():
    num_classes = 10 
    net=  d2l.resnet18(num_classes,3)
    return net 

loss = nn.CrossEntropyLoss(reduction='none')
def train(net,train_iter,valid_iter,num_epochs,lr,wd,devices,lr_period,lr_decay):
    trainer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=wd)
    # 学习率调整器
    scheduler = torch.optim.lr_scheduler.StepLR(trainer,lr_period,lr_decay)
    
    num_batches,timer = len(train_iter),d2l.Timer()
    
    legend = ['train_loss','train_acc']
    if valid_iter is not None:
        legend.append('valid_acc')

    # 注解
    animator = d2l.Animator(xlabel='epoch',xlim=[1,num_batches],legend=legend)

    net = nn.DataParallel(net,devices_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metrics = d2l.Accumulator(3)
        for i ,(features,labels) in enumerate(train_iter):
            timer.start()
            l,acc = d2l.train_batch_ch13(net,features,labels,loss,trainer,devices)
            metrics.add(l,acc,labels.shape[0])
            timer.stop()

            if (i+1)%(num_batches//5)== 0 or i== num_batches -1:
                animator.add(epoch+(i+1)/num_batches,(metrics[0]/metrics[2],metrics[1]/metrics[2],None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net,valid_iter)
            animator.add(epoch+1,(None,None,valid_acc))
        scheduler.step()
    measures = (f'train loss {metrics[0] / metrics[2]:.3f}, '
                f'train acc {metrics[1] / metrics[2]:.3f}')
    
    print(measures + f'\n{metrics[2] * num_epochs / timer.sum():.1f}'f' examples/sec on {str(devices)}')


'''训练和验证模型'''
devices ,num_epochs ,lr,wd = d2l.try_gpu(),20,2e-4,5e-4 
lr_period,lr_decay,net = 4,0.9,get_net()
train(net,train,valid_iter,num_epochs,lr,wd,devices,lr_period,lr_decay)

'''对测试集进行分类并提交结果'''
net = get_net()
preds = []
train(net,train,valid_iter,None,num_epochs,lr,wd,devices,lr_period,lr_decay)

for X,_ in test_iter:
    y_hat = next(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

sorted_idx = list(range(1,len(test_ds)+1))
sorted_idx.sort(key=lambda x:str(x))
df = pd.DataFrame({'id':sorted_idx,'label':preds})

df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)

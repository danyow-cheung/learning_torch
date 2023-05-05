import os 
import pandas as pd 
import torch 
import torchvision
from d2l import torch as d2l 
from torch import nn 
import matplotlib.pyplot as plt 

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')


def read_data_bananas(is_train=True):
    '''读取香蕉检测数据集中的图像和标签'''
    data_dir = d2l.download_extract('banana-detection')
    # csv_fname = os.path.join(data_dir,'bananas_train'if is_trian else 'banans_val','label_csv')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
                             
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images = []
    targets = []
    for img_name,target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir,'bananas_train' if is_train else 'bananas_val','images',f'{img_name}'
        )))
        # 这里target包含(类别,左上角x，左上角y，右下角x，右下角y)
        # 其中所有图像都具有相同的香蕉类(索引为0)
        targets.append(list(target))
    return images,torch.tensor(targets).unsqueeze(1)/256

class BananasDataset(torch.utils.data.Dataset):
    '''一个用于加载香蕉检测数据集的自定义数据集'''
    def __init__(self,is_train):
        self.features,self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' validation examples'))
    def __getitem__(self,idx):
        return(self.features[idx].float(),self.labels[idx])
    
    def __len__(self):
        return len(self.features)

# 定义load_data_bananas函数，为训练集和测试集返回两个数据加载器实例,
def load_data_bananas(batch_size):
    '''加载香蕉检测数据集'''
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),batch_size,shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),batch_size)

    return train_iter,val_iter





def cls_predictor(num_inputs,num_anchors,num_classes):
    return nn.Conv2d(num_inputs,num_anchors*(num_classes +1),kernel_size = 3,padding=1)



# 和类比预测框类似，唯一不同的是，需要为每个锚框预测4个偏移量，而不是q+1类别
def bbox_predictor(num_inputs,num_anchors):
    return nn.Conv2d(num_inputs,num_anchors*4,kernel_size=3,padding=1)

# 多尺度预测
def forward(x,block):
    return block(x)

Y1 =forward(torch.zeros((2,8,20,20)),cls_predictor(8,5,10))

Y2 =forward(torch.zeros((2,16,10,10)),cls_predictor(16,3,10))


def flatten_pred(pred):
    return torch.flatten(pred.permute(0,2,3,1),start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds],dim=1)

# 高和宽减半块
def down_sample_blk(in_channels,out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


# 基本网络块
def base_net():
    blk = []
    num_filters = [3,16,32,64]
    for i in range(len(num_filters)-1):
        blk.append(down_sample_blk(num_filters[i],num_filters[i+1]))

    return nn.Sequential(*blk)


def get_blk(i):
    if i==0:
        blk=base_net()
    elif i==1:
        blk = down_sample_blk(64,128)
    elif i==4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128,128)
    return blk 
    

# 为每个块定义前向传播，
def blk_forward(X,blk,size,ratio,cls_predictor,bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y,sizes=size,ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y,anchors,cls_preds,bbox_preds)




class TinySSD(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(TinySSD,self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64,128,128,128,128]
        
        for i in range(5):
            # 即赋值语句
            setattr(self,f'blk_{i}',get_blk(i))
            setattr(self,f'cls_{i}',cls_predictor(idx_to_in_channels[i],num_anchors,num_classes))
            setattr(self,f'bbox_{i}',bbox_predictor(idx_to_in_channels[i],num_anchors))
            # self.blk_i = get_blk(i)

    def forward(self,X):
        anchors,cls_preds,bbox_preds = [None]*5,[None]*5,[None]*5
        for i in range(5):

            X,anchors[i],cls_preds[i],bbox_preds[i] = blk_forward(
                X,getattr(self,f'blk_{i}'),sizes[i],ratios[i],
                getattr(self,f'cls_{i}'),getattr(self,f'bbox_{i}'))

        anchors = torch.cat(anchors,dim=1)

        cls_preds = concat_preds(cls_preds)
        
        print(cls_preds.shape)# add  这一步已经错落

        cls_preds = cls_preds.reshape(cls_preds.shape[0],-1,self.num_classes+1)

        bbox_preds = concat_preds(bbox_preds)
        return anchors,cls_preds,bbox_preds



def calc_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks):
    batch_size ,num_classes = cls_preds.shape[0],cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1,num_classes),cls_labels.reshape(-1)).reshape(batch_size,-1).mean(dim=1)
    bbox = bbox_loss(bbox_preds*bbox_masks,bbox_labels*bbox_masks).mean(dim=1)
    return cls + bbox 

def cls_eval(cls_preds,cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype)==cls_labels).sum())


def bbox_eval(bbox_preds,bbox_labels,bbox_masks):
    return float((torch.abs((bbox_labels-bbox_preds)*bbox_masks)).sum())




if __name__ =='__main__':
  
    
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],[0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1

    net = TinySSD(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    print('output anchors:', anchors.shape)
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)



    # 小批量计算虽然高效，但它要求每张图像含有相同数量的边界框，以便放在同一个批量中
    batch_size =32 
    train_iter,_ = load_data_bananas(batch_size)
    

    device = d2l.try_gpu()
    net = TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(),lr=0.2,weight_decay=5e-4)
    
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    num_epochs = 2
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['class error','bbox mae'])

    net = net.to(device)
    for epoch in range(num_epochs):
        # 训练精度的和，训练精度的和中示例数
        # 绝对误差和，绝对误差的和中示例数
        metric = d2l.Accumulator(4)
        net.train()
        for features,target in train_iter:
            timer.start()
            trainer.zero_grad()
            X,Y = features,target
            # 生成多尺度的框，为每个框预测类别和偏移量
            anchors,cls_preds ,bbox_preds = net(X)
            # 为每个框标注类别和偏移量
            bbox_labels,bbox_masks,cls_labels = d2l.multibox_target(anchors,Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = calc_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds,cls_labels),cls_labels.numel(),
                        bbox_eval(bbox_preds,bbox_labels,bbox_masks),
                        bbox_labels.numel())

        cls_err,bbox_mae = 1-metric[0]/metric[1],metric[2]/metric[3]
        animator.add(epoch+1,(cls_err,bbox_mae))

    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on'f'{str(device)}')
    
    # 预测目标
    X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
    img = X.squeeze(0).permute(1,2,0).long()
    def predict(X):
        # 使用mutilbox函数，根据xx预测边界框，通过非极大值抑制来移除相似的预测边界框
        net.eval()# 推理模式
        anchors ,cls_preds,bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds,dim=2).permute(0,2,1)
        output = d2l.multibox_detection(cls_probs,bbox_preds,anchors)
        idx = [i for i,row in enumerate(output[0]) if row[0]!=-1]
        return output[0,idx]
    
    output = predict(X)
    print('output',output)
    def display(img,output,threshold):
        d2l.set_figsize((5,5))
        fig = plt.imshow(img)
        for row in output:
            score = float(row[1])
            if score<threshold:
                continue
            h,w = img.shape[0:2]
            bbox =[ row[2:6]*torch.tensor((w,h,w,h),device=row.device)]
            d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    display(img,output.cpu(),threshold=0.9)

import os 
import torchvision
import torch

def read_voc_images(voc_dir,is_trian=True):
    '''读取所有VOC图像并标注'''
    txt_fname = os.path.join(voc_dir,'ImageSets','Segmentation','train.txt' if is_trian else 'val.txt')
    mode = torchvision.io.ImageReadMode.RGB 
    with open(txt_fname,'r') as f:
        images = f.read().split()
    
    features = []
    labels = []
    for i,fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(voc_dir,'JPEGImages',f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir,'SegmentationClass',f'{fname}.png'),mode))
    return features,labels



VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']



  
def voc_color2label():
    '''构建从RGB到VOC类别索引的映射'''
    colormap2label = torch.zeros(256**3,dtype=torch.long)
    for i,colormp in enumerate(VOC_COLORMAP):
        colormap2label[(colormp[0]*256+colormp[1])*256 +colormp[2]]=i 
    return colormap2label


def voc_label_indices(colormap,colormap2label):
    '''将voc标签中的RGB值映射到他们的类别索引'''
    colormap = colormap.permute(1,2,0).numpy().astype('int32')
    idx = ((colormap[:,:,0]*256+colormap[:,:,1]) * 256 +colormap[:,:,2])
    return colormap2label[idx]


def voc_rand_crop(feature,label,height,width):
    '''随机裁剪特征和标签图像'''
    rect = torchvision.transforms.RandomCrop.get_params(
        feature,(height,width)
    )
    feature =  torchvision.transforms.functional.crop(feature,*rect)
    label = torchvision.transforms.functional.crop(label,*rect)
    return feature,label


class VOCSegDataset(torch.utils.data.Dataset):
    '''
    自定义了一个语义分割数据集类VOCSegDataset。
    通过实 现__getitem__函数，我们可以任意访问数据集
    中索引为idx的输入图像及其每个像素的类别索引。
    由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，
    这些样本可以通过自定义的filter函数移除掉。 
    此外，我们还定义了normalize_image函数，从而对输入图像的RGB三个通道的值分别做标准化。
    '''
    def __init__(self,is_train,crop_size,voc_dir):
        self.transform = torchvision.transforms.Normalize( 
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
        self.crop_size = crop_size
        features,labels = read_voc_images(voc_dir,is_trian=is_train)
        self.features = [self.normalize_image(features) for features in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_color2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self,img):
        return self.transform(img.float()/255)
    
    def filtter(self,imgs):
        return [img for img in imgs if (img.shape[1]>=self.crop_size[0] and img.shape[2]>= self.crop_size[1])]
    
    def __getitem__(self,idx):
        feature,label = voc_rand_crop(self.features[idx],self.labels[idx],*self.crop_size)
        return (feature,voc_label_indices(label,self.colormap2label))
    
    def __len__(self):
        return len(self.features)
    

import d2l 
def load_data_voc(batch_size,crop_size):
    '''加载VOC语义分割数据集'''
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True,crop_size,voc_dir),batch_size,
        shuffle=True,drop_last = True,num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False,crop_size,voc_dir),batch_size,
        drop_last=True,num_workers=num_workers
    )
    return train_iter,test_iter

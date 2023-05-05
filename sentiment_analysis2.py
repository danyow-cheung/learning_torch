'''使用卷积神经网络做情感分析'''
import os 
import torch 
from torch import nn 
from d2l import torch as d2l 
batch_size = 64 
train_iter ,test_iter,vocab = d2l.load_data_imdb(batch_size)

def corr1d(X,K):
    w = K.shape[0]
    y = torch.zeros((X.shape[0]-w+1))
    for i in range(y.shape[0]):
        y[i] = (X[i: i + w] * K).sum()

    return y 
def corr1d_multi_in(X,K):
    # 首先，遍历X和K的第0维，然后，把他们加在一起
    return sum(corr1d(x,k) for x,k in zip(X,K))

class TextCNN(nn.Module):
    def __init__(self, vocab_size,embed_size,kernel_size,num_channels,**kwargs) -> None:
        super(TextCNN,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        # 这个嵌入层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size,embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels),2)

        # 最大时间汇聚层没有参数，因此可以共享实例e
        self.pool = nn.AdaptiveAvgPool1d()
        self.relu = nn.ReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c,k in zip(num_channels,kernel_size):
            self.convs.append(nn.Conv1d(2*embed_size,c,k))

    def forward(self,inputs):
        # 沿着向量维度将两个嵌入层连结起来
        # 每个嵌入层的输出形状都是(批量大小，词元数量，词元向量维度)
        embeddings = torch.cat((
            self.embedding(inputs),
            self.constant_embedding(inputs)
        ),dim=2)
        # 根据一维卷积的输入格式，重新排列张量，以便通道作为第二维
        embeddings = embeddings.permute(0,2,1)
        # 每个一维卷积在最大时间汇聚层合并后，获得张量形状（批量大小，通道数，1)
        # 删除最后一个维度并沿通道维度链接
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
        for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))

        return outputs

embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab),embed_size,kernel_sizes,nums_channels)
def init_weights(m):
    if type(m) in (nn.Linear,nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad=False 
# 训练和评估模型
lr = 0.001 
num_epochs = 5 
trainer = torch.optim.Adam(net.parameters(),lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net,train_iter,test_iter,loss,trainer,num_epochs,devices)
 
d2l.predict_sentiment(net,vocab,'this moive is so great ')
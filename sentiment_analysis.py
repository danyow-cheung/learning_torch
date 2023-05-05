import os 
import torch 
from torch import nn 
from d2l import torch as d2l 

d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb','aclImdb')

def read_imdb(data_dir,is_train):
    '''读取imdb评论数据集文本序列和标签'''
    data = []
    labels = []
    for label in ("pos",'neg'):
        folder_name = os.path.join(data_dir,'train' if is_train else 'test',label)

        for file in os.listdir(folder_name):

            with open(os.path.join(folder_name,file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n','')
                data.append(review)
                labels.append(1 if label=='pos' else 0 )
    return data,labels


train_data = read_imdb(data_dir, is_train=True)
'''预处理数据集'''
train_tokens = d2l.tokenize(train_data[0],token='word')
vocab = d2l.Vocab(train_tokens,min_freq=5,reserved_tokens=['<pad>'])
num_steps = 500# 序列长度
train_features = torch.tensor([d2l.truncate_pad(
    vocab[line],num_steps,vocab['<pad>']
) for line in train_tokens])

'''创建数据迭代器'''
train_iter = d2l.load_array((train_features,torch.tensor(train_data[1])),64)
for X,y in train_iter:
    print(X.shape,y.shape)
    break 
print(len(train_iter))

def load_data_imdb(batch_size,num_steps=500):
    '''返回数据迭代器和IMDb评论数据集的词表'''
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir,True)
    test_data = read_imdb(data_dir,False)
    train_tokens = d2l.tokenize(train_data[0],token='word')
    test_tokens = d2l.tokenize(test_data[0],token='word')
    vocab = d2l.Vocab(train_tokens,min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(vocab[line],num_steps,vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(vocab[line],num_steps,vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((
        train_features,torch.tensor(train_data[1])
    ),batch_size)
    test_iter = d2l.load_array((test_features,torch.tensor(test_data[1])),batch_size,is_train=False)
    return train_iter,test_iter,vocab 


batch_size = 64 
train_iter,test_iter,vocab = d2l.load_data_imdb(batch_size)
'''使用循环神经网络表示单个文本'''

class BiRNN(nn.Module):
    def __init__(self, vocab_size,embed_size,num_hiddens,num_layers,**kwargs) -> None:
        super(BiRNN,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size,num_hiddens,num_layers=num_layers,bidrectional=True)
        self.decoder = nn.Linear(4*num_hiddens,2)

    def forward(self,inputs):
        # inputs的形状是(批量大小，时间步数)
        # 因为⻓短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为(时间步数，批量大小，词向量维度)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态
        # outputs的形状是(时间步数，批量大小，2*隐藏单元数)
        outputs,_ = self.encoder(embeddings)
        # 连结初识和最终时间步的隐状态，作为全链接层的输入
        encoding = torch.cat(outputs[0],outputs[-1],dim=1)
        outs = self.decoder(encoding)
        return outs
embed_size = 100 
num_hiddens = 100 
num_layers = 2
devies = d2l.try_all_gpus()
net = BiRNN(len(vocab),embed_size,num_hiddens,num_layers)
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m)==nn.LSTM:
        for param in m._flat_weights_names:
            if 'weights' in param:
                nn.init.xavier_uniform(m._parameters[param])

net.apply(init_weights)

'''加载预训练的词向量'''
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad=False 

'''训练和评估模型'''
lr = 0.01 
num_epochs = 5 
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net,train_iter,test_iter,loss,trainer,num_epochs)

def predict_sentiment(net,vocab,sequence):
    '''预测文本序列的情感'''
    sequence - torch.tensor(vocab[sequence.split()],device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1,-1)),dim=1)
    return 'positive' if label==1 else 'negative'

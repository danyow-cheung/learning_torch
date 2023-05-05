import os 
import re 
import torch
from torch import nn 
from d2l import torch as d2l 
from torch.nn import functional as F
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4'

)

data_dir = d2l.download_extract('SNLI')
'''读取数据集'''
def read_snli(data_dir,is_train):
    '''将SNLI数据集解析为前提，假设和标签'''
    def extract_text(s):
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替代多个或连续空格键
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    
    label_set = {'entailment':0,'contradiction':1,'neutral':2,}
    file_name = os.path.join(data_dir,'snli_1.0_train.txt' if is_train else 'snli_1.0.test.txt')
    with open(file_name,'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]

    permises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypothess = [extract_text(row[2]) for row in rows if row[0] in label_set]

    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return permises,hypothess,labels

train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print(f'前提={x0} 假设={x1} 标签={y}')

test_data = read_snli(data_dir,is_train=False)
for data in [train_data,test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])


'''定义用于加载数据集的类'''
class SNLIDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,num_steps,vocab=None):
        self.num_steps = num_steps 
        all_permise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_hypothesis_tokens+all_hypothesis_tokens,min_freq=5,reserved_tokens=['<pad>'])

        else:
            self.vocab = vocab
        self.permises = self._pad(all_permise_tokens)
        self.hypothess = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')
    
    def _pad(self,lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line],self.num_steps,self.vocab['<pad>']
        ) for line in lines 
        ])
    
    def __getitem__(self,idx):
        return (self.permises[idx] ,self.hypothess[idx]),self.labels[idx]
    
    def __len__(self):
        return len(self.permises)
    
def load_data_snli(batch_size,num_steps=50):
    '''下载数据集并返回数据迭代器和词表'''
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir,True)
    test_data = read_snli(data_dir,False)
    train_set = SNLIDataset(train_data,num_steps)
    test_set = SNLIDataset(test_data,num_steps,train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set,batch_size,shuffle=True,num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set,batch_size,shuffle=False,num_workers=num_workers)

    return train_iter,test_iter,train_set.vocab

train_iter, test_iter, vocab = load_data_snli(128, 50)


def mlp(num_inputs,num_hiddens,flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs,num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens,num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))

    return nn.Sequential(*net)

class Attend(nn.Module):
    def __init__(self,num_inputs,num_hiddens,**kwargs):
        super(Attend,self).__init__(**kwargs)
        self.f = mlp(num_inputs,num_hiddens,flatten=False)
    
    def forward(self,A,B):
        f_A = self.f(A)
        f_B = self.f(B)
        e = torch.bmm(f_A,f_B.permute(0,2,1))
        beta = torch.bmm(F.softmax(e,dim=1),B)
        alpha = torch.bmm(F.softmax(e.permute(0,2,1),dim=-1),A)
        return beta,alpha
    

class Compare(nn.Module):
    def __init__(self,num_inputs,num_hiddens,**kwargs):
        super(Compare,self).__init__(**kwargs)
        self.g = mlp(num_inputs,num_hiddens,flatten=False)

    def forward(self,A,B,beta,alpha):
        V_A = self.g(torch.cat([A,beta],dim=2))
        V_B = self.g(torch.cat([B,alpha],dim=2))
        return V_A,V_B
    

class Aggregate(nn.Module):
    def __init__(self,num_inputs,num_hiddens,num_outputs,**kwargs):
        super(Aggregate,self).__init__(**kwargs)
        self.h = mlp(num_inputs,num_hiddens,flatten=True)
        self.linear = nn.Linear(num_hiddens,num_outputs)

    def forward(self,V_A,V_B):
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        Y_hat = self.linear(self.h(torch.cat([V_A,V_B],dim=1)))
        return Y_hat
    

class DecomposableAttention(nn.Module):
    def __init__(self,vocab,embed_size,num_hiddens,num_inputs_attend=100,num_inputs_compare=200,num_inputs_agg=400,**kwargs):
        super(DecomposableAttention,self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab),embed_size)
        self.attend = Attend(num_inputs_attend,num_hiddens)
        self.compare = Compare(num_inputs_compare,num_hiddens)
        # 有3种可能输出
        self.aggreate = Aggregate(num_inputs_agg,num_hiddens,num_outputs=3)

    def forward(self,X):
        premises,hypotheses = X 
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta,alpha = self.attend(A,B)
        V_A,V_B = self.compare(A,B,beta,alpha)
        Y_hat = self.aggreate(V_A,V_B)
        return Y_hat

batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
embed_size,num_hiddens,devices = 100,200,d2l.try_all_gpus()
net = DecomposableAttention(vocab,embed_size,num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)

lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,devices)
'''使用模型'''
def predict_snli(net,vocab,premise,hypothesis):
    '''预测前提和假设之间的逻辑关系'''
    net.eval()
    premise = torch.tensor(vocab[premise],device=d2l.try_all_gpus())
    hypothesis = torch.tensor(vocab[hypothesis],device=d2l.try_gpu())
    label = torch.argmax(
        net([premise.reshape((1,-1)),
             hypothesis.reshape((1,-1))]),dim=1
    )
    return 'entailment' if label == 0 else 'contradiction' if label == 1  else 'neutral'    



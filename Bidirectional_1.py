'''3.19
双向循环神经网络的错误应用
'''
import torch 
from torch import nn 
from d2l import torch as d2l 

# 加载数据
batch_size = 32 
num_steps = 35 

device = d2l.try_gpu(0)

train_iter,vocab = d2l.load_data_time_machine(batch_size,num_steps)
# 通过设置‘bidirective=True’双向定义LSTM模型
vocab_size,num_hiddens,num_layers = len(vocab),256,2 
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs,num_hiddens,num_layers,bidirectional=True)
model = d2l.RNNModel(lstm_layer,len(vocab))
model = model.to(device)

# 训练模型
num_epochs = 500 
lr = 1 
d2l.train_ch8(model,train_iter,vocab,lr,num_epochs,device)

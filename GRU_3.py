
import torch
from torch import nn 
from d2l import torch as d2l 
'''深度循环神经网络简洁实现'''

batch_size = 32 
num_steps = 35 

train_iter,vocab = d2l.load_data_time_machine(batch_size,num_steps)
vocab_size = len(vocab)
num_hiddens = 256 
num_layers = 2 
num_inputs = vocab_size

device = d2l.try_gpu(0)
# 定义了2层lstm单元
lstm_layer = nn.LSTM(num_inputs,num_hiddens,num_layers)
model = d2l.RNNModel(lstm_layer,len(vocab))
model = model.to(device)

num_epochs = 500 
lr = 2 
d2l.train_ch8(model,train_iter,vocab,lr*1.0,num_epochs,device)

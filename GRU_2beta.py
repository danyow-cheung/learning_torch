import torch
from torch import nn 
from d2l import torch as d2l 
'''LSTM简洁实现'''
batch_size = 32 
num_steps = 35 

train_iter,vocab = d2l.load_data_time_machine(batch_size,num_steps)

'''3. 训练和预测'''
vocab_size = len(vocab)
num_hiddens = 256 
device = d2l.try_gpu(0)
num_epochs = 500 
lr =  1

num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs,num_hiddens)
model = d2l.RNNModel(lstm_layer,len(vocab))
model = model.to(device)
d2l.train_ch8(model,train_iter,vocab,lr,num_epochs,device)

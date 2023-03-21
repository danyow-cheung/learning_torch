import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


class Seq2Seq_Encoder(d2l.Encoder):
	def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
		super(Seq2Seq_Encoder,self).__init__(**kwargs)
		# 嵌入层
		self.embedding = nn.Embedding(vocab_size,embed_size)
		self.rnn = nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)

	def forward(self,X,state):
		# 输出X的形状,(batch_size,num_steps,embed_size)
		X = self.embedding(X)
		# 在循环神经网络中，第一个轴对应时间步
		X = X.permut(1,0,2)
		# 如果未提及状态，则默认为0
		output,state = self.rnn(x)

		# output形状(num_steps,batch_size,num_hiddens)
		# state形状(num_layers,batch_size,num_hiddens)
		return output,state


encoder = Seq2Seq_Encoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layer=2)
encoder.eval()
X = torch.zeros((4,7),dtype=torch.long)
output,state = encoder(X)
print(output.shape)

'''指定长度可变的序列作为编码的输入X，任何继承与这个Encoder基类的模型将完成'''
from torch import nn 

class Decoder(nn.Module):
	'''编码器-解码器架构的基本编码器接口'''
	def __init__(self,**kwargs):
		super(Decoder,self).__init__(**kwargs)

	def init_state(self,enc_outputs,*args):
		raise NotImplementedError 

	def forward(self,X,state):
		raise NotImplementedError 

		


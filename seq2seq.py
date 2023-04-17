import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

'''编码器'''

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
		X = X.permute(1,0,2)
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

'''解码器'''
class Seq2Seq_Decoder(d2l.Decoder):
	def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
		super(Seq2Seq_Decoder,self).__init__(**kwargs)
		self.embedding = nn.Embedding(vocab_size,embed_size)
		self.rnn = nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)

		self.dense = nn.Linear(num_hiddens,vocab_size)

	def init_state(self,enc_outputs,*args):
		return enc_outputs[1]

	def forward(self,X,state):
		# 输出X的尺寸(batch_size,num_steps,embed_size)
		X= self.embedding(X).permute(1,0,2)
		# 广播context，使其具有与X相同的num_steps
		context = state[-1].repeat(X.shape[0],1,1)
		X_and_context = torch.cat((X,context),2)
		output,state = self.rnn(X_and_context,state)
		output = self.dense(output).permute(1,0,2)
		# output的形状(batch_size,num_steps,vocab_size)
		# state的形状(num_layers,batch_size,num_hiddens)
		return output,state 
'''使用相同参数来实例化解码器'''

decoder = Seq2Seq_Decoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
decoder.eval()

state = decoder.init_state(encoder())
output,state = decoder(X,state)
print(output.shape,state.shape)

'''在每个时间步，decoder预测词元的概率分布，使用softmax函数来获得，并通过计算交叉熵损失损失函数进行优化'''
# 使用sequence_mask函数通过零值化来屏蔽不相关的项，方便后面任何不相关预测的计算都是零的乘积，增快训练速度
def sequence_mask(X,valid_len,value=0):
	'''在序列中屏蔽不相关的项'''
	
	maxlen = X.size(1)

	mask = torch.arange((maxlen),dtype=torch.float32,device=X.device)[None,:]<valid_len[:,None]

	X[-mask]=value 
	return X 

# 测试sequence_mask函数
X = torch.tensor([[1,2,3],[4,5,6]])
sequence_mask(X,torch.tensor([1,2]))

# 通过扩展softmax交叉熵损失函数来掩遮不相关的预测，最初所有预测词元的掩码都设置为1，一旦给定了有效长度，与填充词元对应的掩码都设置为0
# 将所有词元的损失乘掩码，以过滤损失中填充词元产生的不相关预测
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
	'''带遮掩的softmax交叉熵损失函数'''
	# pred形状			(batch_size,num_steps,vocab_size)
	# label形状			(batch_size,num_steps)
	# valid_len形状		(batch_size,)
	def forward(self,pred,label,valid_len):
		weights = torch.ones_like(label)
		weights = sequence_mask(weights,valid_len)
		self.reduction = 'none'
		unweighted_loss = super(MaskedSoftmaxCELoss,self).forward(pred.premute(0,2,1),label)
		weighted_loss = (unweighted_loss*weights).mean(dim=1)
		return weighted_loss
# 创建三个相同序列来进行代码健全检查
loss = MaskedSoftmaxCELoss()
loss(torch.ones(3,4,10),torch.ones((3,4),dtype=torch.long),torch.tensor([4,2,0]))

'''
特定的序列开始词元和原始的输出序列拼接在一起作为解码器的输入，这被称为强制教学，因为原始的输出序列被送入解码器
-------------
将上一个时间步预测得到的词元作为解码器的当前输入
'''
def train_seq2seq(net,data_iter,lr,num_epochs,tgt_vocab,device):
	'''训练seq2seq模型'''
	def xavier_init_weights(m):
		if type(m)==nn.Linear:
			nn.init.xavier_uniform_(m.weight)
		if type(m)==nn.GRU:
			for param in m._flat_weights_names:
				if 'weights' in param:
					nn.init.xavier_uniform_(m._parameters[param])

	nn.apply(xavier_init_weights)
	net.to(device)
	optimizer = torch.optim.Adam(net.parameters(),lr=lr)
	loss = MaskedSoftmaxCELoss()
	net.train()
	animator = d2l.Animator(xlabel='epoch',ylabel='loss',xlim=[10,num_epochs])
	for epoch in range(num_epochs):
		timer = d2l.Timer()
		metric = d2l.Accumulator() # 训练损失总和，词元数量

		for batch in data_iter:
			optimizer.zero_grad()
			X,X_valid_len,Y,Y_valid_len = [X.to(device) for x in batch]
			bos = torch.tensor([tgt_vocab['<bos>']] *Y.shape[0],device=device).reshape(-1,1)
			dec_input = torch.cat([bos,Y[:,:,-1]],1) # 强制教学
			Y_hat,_ = net(X,dec_input,X_valid_len)
			l = loss(Y_hat,Y_valid_len)

			l.sum().backward()  # 损失函数的标量进行“反向传播”
			d2l.grad_clipping(net,l)
			num_tokens = Y_valid_len.sum()
			optimizer.step()
			with torch.no_grad():
				metric.add(l,sum(),num_tokens)
		if (epoch+1)%10==0:
			animator.add(epoch+1,(metric[0]/metric[[1],]))
	print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
       f'tokens/sec on {str(device)}')
	
# 在机器翻译数据集上，创建和训练一个循环神经网络用于序列到序列的学习
embed_size = 32 
num_hiddens = 32 
num_layers = 2 
dropout =0.1 

batch_size = 64 
num_steps = 10 

lr = 0.005 
num_epochs = 300 
device = d2l.try_gpu(0)

train_iter ,src_vocab ,tgt_vocab = d2l.load_data_nmt(batch_size,num_steps)
encoder = Seq2Seq_Encoder(len(src_vocab),embed_size,num_hiddens,num_layers,dropout)
decoder = Seq2Seq_Decoder(len(tgt_vocab),embed_size,num_hiddens,num_layers,dropout)
net = d2l.EncoderDecoder(encoder,decoder)
train_seq2seq(net,train_iter,num_epochs,tgt_vocab,device)


# 预测
def predict_seq2seq(net,src_sentence,src_vocab,tgt_vocab,num_steps,device,save_attention_weights=False):
	'''序列到序列模型的预测'''
	# 在预测时将net设置为评估模式
	net.eval()
	src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
	enc_valid_len = torch.tensor([len(src_tokens)],device=device)
	src_tokens = d2l.truncate_pad(src_tokens,num_steps,src_vocab['<pad>'])
	# 添加批量轴
	dec_X = torch.unsqueeze(
		torch.tensor(src_tokens,dtype=torch.long,device=device),dim=0
	)
	output_seq,attention_weight_seq = [],[]
	for _ in range(num_steps):
		Y,dec_state = net.decode(dec_X,dec_state)
		# 使用预测概率最高的词元，作为解答器在下一时间步的输入
		dec_x = Y.argmax(dim=2)
		pred = dec_x.squeeze(dim=0).type(torch.int32).item()
		# 保存注意力权重
		if save_attention_weights:
			attention_weight_seq.append(net.decoder.attention_weights)
		
		# 一旦序列结束词元被预测，输出序列生成完成
		if pred==tgt_vocab['<eos>']:
			break
		output_seq.append(pred)

	return "".join(tgt_vocab.to_tokens(output_seq)),attention_weight_seq

'''预测序列的评估--BLEU方法来作为机器翻译结果的参数，根据定义当预测序列与标签序列完全相同，BLEU为1，
当n元语法越越匹配难度越大，所以BLEU为更长n元语法的精确度分配更大的权重
'''
def bleu(pred_seq,label_seq,k):
	'''计算BLEU'''
	pred_tokens,label_tokens = pred_seq.split(' '),label_seq.split(' ')
	len_pred = len(pred_tokens)
	len_label = len(label_tokens)
	score = math.exp(min(0,1-len_label/len_pred))
	for n in range(1,k+1):
		num_matches =0
		label_subs =collections.defaultdict(int)
		for i in range(len_label-n+1):
			label_subs[' '.join(label_tokens[i:i+n])]+=1 
		for i in range(len_pred-n+1):
			if label_subs[' '.join(pred_tokens[i:i+n])]>0:
				num_matches+=1 
				label_subs[' '.join(pred_tokens[i:i+n])]-=1 
		score *= math.pow(num_matches/(len_pred-n+1),math.pow(0.5,n))
	return score
# 测试刚才的bleu函数
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']

fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng,fra in zip(engs,fras):
	translation,attention_weight_seq = predict_seq2seq(
		net,eng,src_vocab,tgt_vocab,num_steps,device
	)
	print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
	
 
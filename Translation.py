'''3.19 《机器翻译与数据集》
步骤
1. 下载和预处理数据集
2. 词元化
3. 词表
4. 加载数据集 
5. 训练模型
'''
import os 
import torch 
from d2l import torch as d2l 

# step 1:
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL+'fra-eng.zip','94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
	'''加载英语-法院数据集'''
	data_dir = d2l.download_extract('fra-eng')
	with open(os.path.join(data_dir,'fra.txt'),"r",encoding='utf-8') as f:
		return f.read()

raw_text = read_data_nmt()
print(raw_text[:100])

'''下载数据集后，原始文本数据经过预处理步骤
1. 空格代替不间断空格
2. 使用小写字母代替大写字母
3. 在单词和标点之间添加空格
'''
def preprocess_nmt(text):
	def no_space(char,prev_char):
		return char in set(',.!?') and prev_char != ' '

	# 使用空格代替不间断空格
	# 使用小写字母代替大写字母
	text = text.replace('\u202f',' ').replace('\xa0',' ').lower()

	# 在单词和标点之间添加空格
	out = [" " + char if i>0 and no_space(char,text[i-1]) else char for i,char in enumerate(text)]
	return "".join(out)

text = preprocess_nmt(raw_text)
print(text[:100])

def tokenize_nmt(text,num_examples=None):
	'''词元化：英语-法语 数据数据集'''
	source = []
	target = []
	for i,line in enumerate(text.split('\n')):
		if num_examples and i>num_examples:
			break 
		parts = line.split('\t')
		if len(parts)==2:
			source.append(parts[0].split(' '))
			target.append(parts[1].split(' '))
	return source,target


source,target = tokenize_nmt(text)
print("词元后的示例")
print(source[:6])
print(target[:6])


'''绘制文本序列所包含的词元数量的直方图'''
def show_list_len_pair_hist(legend,xlabel,ylabel,xlist,ylist):
	d2l.set_figsize()
	_,_,patches = d2l.plt.hist(
		[[len(l) for l in xlist],[len(l) for l in ylist]])
	
	d2l.plt.xlabel(xlabel)
	d2l.plt.ylabel(ylabel)
	for patch in patches[-1].patches:
		patch.set_hatch('/')
	d2l.plt.legend(legend)
show_list_len_pair_hist(['source','target'],'count',source,target)

'''词表'''
src_vocab = d2l.Vocab(source,min_freq=2,reserved_tokens = ['<pad>','<bos>','eos'])

print(len(src_vocab))

'''通过截断和填充方式实现一次只处理一个小批量的文本序列'''
def truncate_pad(line,num_steps,padding_token):
	if len(line)>num_steps:
		return line[:num_steps]
	return line + [padding_token]*(num_steps-len(line)) #填充

truncate_pad(src_vocab[source[0]],10,src_vocab['<pad>'])

'''定义一个函数，将文本序列转换为小批量数据集用于训练'''
def build_array_nmt(lines,vocab,num_steps):
	'''将机器翻译的文本序列转换成小批量'''
	lines =[vocab[l] for l in lines]
	lines = [l+[vocab['<eos>']] for l in lines]
	array = torch.tensor([truncate_pad( l,num_steps,vocab['<pad>']) for l in lines])
	valid_len = (array!=vocab['<pad>']).type(torch.int32).sum(1)
	return array,valid_len

def load_data_nmt(batch_size,num_steps,num_examples=600):
	'''返回翻译数据集的迭代器和词表'''
	text = preprocess_nmt(read_data_nmt())
	source,target = tokenize_nmt(text,num_examples)
	src_vocab = d2l.Vocab(source,min_freq=2,reserved_tokens=['<pad>','<bos>','<eos>'])
	tgt_vocab = d2l.Vocab(target,min_freq=2,reserved_tokens=['<pad>','<bos>','<eos>'])

	src_array ,src_valid_len = build_array_nmt(source,src_vocab,num_steps)
	tgt_array ,tgt_valid_len = build_array_nmt(target,tgt_vocab,num_steps)

	data_arrays = (src_array,src_valid_len,tgt_array,tgt_valid_len)
	data_iter = d2l.load_array(data_arrays,batch_size)
	return data_iter,src_vocab,tgt_vocab

# 示范数据集中第一个小批量数据
train_iter,src_vocab,tgt_vocab = load_data_nmt(batch_size=2,num_steps=8)
for X,X_valid_len,Y,Y_valid_len in train_iter:
	print('X:', X.type(torch.int32))
	print('X的有效⻓度:', X_valid_len)
	print('Y:', Y.type(torch.int32))
	print('Y的有效⻓度:', Y_valid_len)
	break 

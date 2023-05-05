import torch 
import os 
import random
import math 
from d2l import torch as d2l 

'''读取数据集'''
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL+'ptb.zip','319d85e578af0cdc590547f26231e4e31cdf1e42')

def read_ptb():
    '''将PTB数据集加载到文本行的列表中'''
    data_dir = d2l.download_extract('ptb')
    # 读取训练数据集
    with open(os.path.join(data_dir,'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
print('sentences number',len(sentences))

'''在读取数据集后，为语料库构建一个词表，其中出现次数少于10次的任何单词都有<unk>词元替换，'''
vocab = d2l.Vocab(sentences,min_freq=10)
print(f'vocab size',len(vocab))

def subsample(sentences,vocab):
    '''下采样高频词'''
    # 排除未知词元
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())
    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return (random.uniform(0,1)<math.sqrt(1e-4/counter[token] * num_tokens))
    
    return ([[token for token in line if keep(token)] for line in sentences],counter)

subsampled ,counter = subsample(sentences,vocab)
d2l.show_list_len_pair_hist(
    ['origin','subsampled'],# tokens per sentence 
    'count',sentences,subsampled
)

'''中心词和上下文词的提取'''
def get_centers_and_contexts(corpus,max_window_size):
    '''返回跳元模型中的中心词和上下文词'''
    centers = []
    contexts = []
    for line in corpus:
        # 要形成，‘中心词-上下文词’对，每个句子至少需要元有2个词
        if len(line)<2:
            continue
        centers += line 
        for i in range(len(line)):
            window_size = random.randint(1,max_window_size)
            indices = list(range(max(0,i-window_size),min(len(line),i+1+window_size)))
            # 上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers,contexts


'''负采样'''
class RandomGenerator:
    '''根据n个采样权重在{1,...n}中随机抽取'''
    def __init__(self,sampling_weights) -> None:
        self.population = list(range(1,len(sampling_weights)+1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0 

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(
                self.population,self.sampling_weights,k=10000
            )
            self.i = 0
        self.i +=1 
        return self.candidates[self.i-1]

corpus = [vocab[line] for line in subsampled]
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)


def get_negatives(all_contexts,vocab,counter,k):
    '''返回负采样中的噪声词'''
    # 索引为1，2，（索引0室词表中排除的未知标记)
    sample_weights = [counter[vocab.to_tokens(i)] **0.75 for i in range(1,len(vocab))]
    all_negatives ,generator = [],RandomGenerator(sample_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts)*k:
            neg  = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)

        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts,vocab,counter ,5)

'''小批量加载训练实例'''
def batchify(data):
    '''返回带有负采样的跳元模型的小批量样本'''
    max_len = max(len(c)+len(n) for _,c,n in data) 
    centers ,contexts_negatives,masks,labels = [],[],[],[]
    for center ,context ,negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative+[0]*(max_len -cur_len)]

        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1,1)),torch.tensor(contexts_negatives),torch.tensor(masks),torch.tensor(labels))


def load_data_ptb(batch_size,max_window_size,num_noise_words):
    '''下载·PTB数据集，然后将其加载到内存中'''
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences,min_freq=10)
    subsampled ,counter = subsample(sentences,vocab)
    corpus =[vocab[line] for line in subsampled]

    all_centers,all_contexts = get_centers_and_contexts(
        corpus,max_window_size,
    )

    all_negatives =  get_negatives(all_contexts,vocab,counter,num_noise_words)


    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self,centers,contexts,negatives):
            assert len(centers)==len(contexts)==len(negatives)
            self.centers = centers 
            self.contexts = contexts
            self.nagatives = negatives

        def __getitem(self,index):
            return (self.centers[index],self.contexts[index],self.nagatives[index])
        
        def __len__(self):
            return len(self.conters)
        
    dataset = PTBDataset(all_centers,all_contexts,all_negatives)
    data_iter = torch.utils.data.DataLoader(
        dataset,batch_size,shuffle=True,
        collate_fn=batchify,num_worker=num_workers
    )
    return data_iter,vocab


data_iter, vocab = load_data_ptb(512, 5, 5)




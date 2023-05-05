import os 
import torch
from torch import nn 
from d2l import torch as d2l 

'''输入表示'''
def get_tokens_and_segments(tokens_a,tokens_b=None):
    '''获取输入序列的词元和片段索引'''
    tokens = ['<cls>']+tokens_a+['<sep>']
    # 0和1分别标记片段A和B
    segments = [0]*(len(tokens_a)+2)

    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1]*(len(tokens_b)+1)

    return tokens,segments

class BERTEncoder(nn.Module):
    '''BERT编码器'''
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,
                 ffn_num_hiddens,num_heads,num_layer,dropout,max_len=1000,
                 key_size=768,query_size=768,value_size = 768,**kwargs):
        
        super(BERTEncoder,self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size,num_hiddens)
        self.segment_embedding = nn.Embedding(2,num_hiddens)

    def forward(self,tokens,segments,valid_lens):
        # 在以下代码段中，x的形状保持不变（批量大小，最大序列长度，num_hiddens）
        X= self.token_embedding(tokens)+self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:,:X.shape[1],:]
        for blk in self.blks:
            X = blk(X,valid_lens)
        return X 
    
vocab_size  = 10000 
num_hiddens = 768 
ffn_num_hiddens = 1024 
num_heads = 4 

norm_shape =[768]
ffn_num_input = 768 
num_layers = 2 
dropout = 0.2 
encoder = BERTEncoder(vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout=)

tokens = torch.randint(0,vocab_size,(2,8))
segments = torch.tensor([[0,0,0,0,1,1,1,1],[0,0,0,1,1,1,1,1]])
encoded_x = encoder(tokens,segments,None)
encoded_x.shape 

class MaskLM(nn.Module):
    '''BERT的掩蔽语言模型任务'''
    def __init__(self,vocab_size,num_hiddens,num_inputs=768,**kwargs):
        super(MaskLM,self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs,num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens,vocab_size)
        )

    def forward(self,X,pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)

        batch_size = X.shape[0]
        batch_idx = torch.arange(0,batch_size)

        batch_idx = torch.repeat_interleave(batch_idx,num_pred_positions)
        masked_X = X[batch_idx,pred_positions]
        masked_X = masked_X.reshape((batch_size,num_pred_positions,-1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
    

mlm = MaskLM(vocab_size, num_hiddens)
'''下一句预测'''
class NextSentencePred(nn.Module):
    def __init__(self,num_inputs,**kwargs):
        super(NextSentencePred,self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs,2)

    def forward(self,X):
        # x的形状（batch_size,num_hiddens)
        return self.output(X)


class BERTModel(nn.Module):
    '''BERT模型'''
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,
                 ffn_num_hiddens,num_heads,num_layers,dropout,max_len = 1000
                 key_size=768,value_size = 768,
                 hid_in_features = 768,
                 mlm_in_features = 768,
                 nsp_in_features = 768,) -> None:
        super(BERTModel,self).__init__()
        self.encoder = BERTEncoder(
            vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,
            num_heads,num_layers,dropout,max_len=max_len,key_size=key_size,
            query_size=query_size,value_size=value_size)
        
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features,num_hiddens),
            nn.Tanh()
        )

        self.mlm = MaskLM(vocab_size,num_hiddens,mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self,tokens,segments,valid_lens= None,pred_positions=None):
        encoded_x = self.encoder(tokens,segments,valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_x,pred_positions)
        else:
            mlm_Y_hat = None 
        # 用于下一句预测的多层感知机分类器的隐藏层，0是'<cls>'标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_x[:,0,:]))
        return encoded_x,mlm_Y_hat,nsp_Y_hat
    
    
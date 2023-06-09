import torch
import math 
from torch import nn 
from d2l import torch as d2l 

class PositionWiseFFN(nn.Module):
    '''
    基於位置的前饋網絡
    '''
    def __init__(self, ffn_num_input,ffn_num_hiddens,ffn_num_outputs, **kwargs) -> None:
        super(PositionWiseFFN,self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens,ffn_num_outputs)
    
    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))
    

class AddNorm(nn.Module):
    '''
    殘差連結後進行層規劃化
    '''
    def __init__(self,normalized_shape,dropout,**kwargs) -> None:
        super(AddNorm,self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self,X,Y):
        return self.ln(self.dropout(Y)+X)


class EncoderBlock(nn.Module):
    '''
    Transformer編碼器快
    '''
    def __init__(self,
                 key_size,query_size,value_size,
                 num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,
                 num_heads,dropout,use_bias =False,
                 **kwargs) -> None:
        super(EncoderBlock,self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size,query_size,value_size,num_hiddens,num_heads,dropout,dropout
        )
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input,ffn_num_hiddens,num_hiddens
        )
        self.addnorm2 = AddNorm(norm_shape,dropout)
    def forward(self,X,valid_lens):
        Y = self.addnorm1(X,self.attention(X,X,X,valid_lens))
        return self.addnorm2(Y,self.ffn(Y))



class TransformerEncoder(d2l.Encoder):
    '''Transformer編碼器'''
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,
                 ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias=False,
                   **kwargs):
        super(TransformerEncoder,self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens,dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block'+str(i),
                                 EncoderBlock(key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,use_bias))
    def forward(self, X,valid_lens, *args):
        # 因為位置編碼在（-1，1）
        # 因此嵌入值乘嵌入維度的平方根進行縮放
        # 然後再與位置編碼相加
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights = [None]*len(self.blks)
        for i,blk in enumerate(self.blks):
            X = blk(X,valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X 

class DecoderBlock(nn.Module):
    '''解碼器的第i個塊'''
    def __init__(self, key_size,query_size,value_size,num_hiddens,
                 norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,i, **kwargs) -> None:
        super(DecoderBlock,self).__init__(**kwargs)
        self.i = i 
        self.attention1 = d2l.MultiHeadAttention(
            key_size,query_size,value_size,num_hiddens,num_heads,dropout
        )
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size,query_size,value_size,num_hiddens,num_heads,dropout
        )
        self.addnorm2 = AddNorm(norm_shape,dropout)
        self.ffn = PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm3 = AddNorm(norm_shape,dropout)
    
    def forward(self,X,state):
        enc_outputs,enc_valid_lens = state[0],state[1]
        # 訓練階段，輸出序列的所有詞元都在同一時間處理
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X 
        else:
            key_values = torch.cat((state[2][self.i],X),axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size,num_steps ,_ = X.shape 
            # dec_valid_lens的開頭(batch_size,num_steps)
            # 其中每一行是[1,2,...num_steps]
            dec_valid_lens = torch.arange(
                1,num_steps+1
            ).repeat(batch_size,1)
        else:
            dec_valid_lens = None
        # 自注意力
        X2  = self.attention1(X,key_values,key_values,dec_valid_lens)
        Y = self.addnorm1(X,X2)
        # 編碼器-解碼器注意力
        # enc_outputs的開頭(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Z = self.addnorm2(Y,Y2)
        return self.addnorm3(Z,self.ffn(Z)),state
    

class TransformerDecoder(d2l.AttentionDecoder):
    '''通过一个全链接层计算所有vocab_size个可能的输出词元的预测值'''
    def __init__(self,vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,
                  num_heads,num_layers,dropout,**kwargs):
        super(TransformerDecoder,self).__init__(**kwargs)

        self.num_hiddens = num_hiddens
        self.num_layers = num_heads
        self.embedding = nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens,dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block'+str(i),
                                DecoderBlock(key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,dropout,i))
        self.dense = nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,enc_valid_lens,*args):
        return [enc_outputs,enc_valid_lens,[None]*self.num_layers]
    
    def forward(self,X,state):
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights = [[None]*len(self.blks) for _ in range(2)]
        for i,blk in enumerate(self.blks):
            X,state = blk(X,state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # 解码器-解码器 自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X),state

    @property
    def attention_weights(self):
        return self.attention_weights
    

if __name__=='__main__':
    # 训练
    num_hiddens = 32 
    num_layers = 2 
    dropout=0.1
    batch_size = 64 
    num_steps = 10 
    lr = 0.005 
    num_epochs = 200 
    device = 'cpu'
    ffn_num_input = 32 
    ffn_num_hiddens = 64 
    num_heads = 4 
    key_size = 32 
    query_size = 32 
    value_size = 32 
    norm_shape = [32]
    train_iter ,src_vocab ,tgt_vocab = d2l.load_data_nmt(batch_size,num_steps)
    encoder = TransformerEncoder(
        len(src_vocab),key_size,query_size,value_size,num_hiddens,
        norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,
        num_layers,dropout
    )
    decoder = TransformerDecoder(
        len(tgt_vocab),key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,
        num_layers,dropout
    )
    net = d2l.EncoderDecoder(encoder,decoder)
    d2l.train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device)
    # 后续暂不进行
    # 后续内容在书本451
    
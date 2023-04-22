import torch
import math 
from torch import nn 
from d2l import torch as d2l 

class MultiHeadAttention(nn.Module):
    '''
    選擇縮放點積注意力作為每一頭注意力頭
    '''
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias =False, *args, **kwargs) -> None:
        super(MultiHeadAttention,self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_o = nn.Linear(num_hiddens,num_hiddens,bias=bias)
    

    
    def forward(self,queries,keys,values,valid_lens):
        # queries,keys,values的形狀
        # (batch_size,查詢或者‘鍵-值’對的個數，num_hiddens)
        # valid_lens 形狀
        # （batch_size,）或者(batch_size,查詢的個數)
        # 經過變換後,輸出的queries,keys,values的形狀
        # (batch_size*num_heads,查詢或者‘鍵-值’對個數, num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries),self.num_heads)
        keys = transpose_qkv(self.W_k(keys),self.num_heads)
        values = transpose_qkv(self.W_v(values),self.num_heads)
        if valid_lens.all():
            # 在軸0，將第一項（標量或矢量）複製num_heads
            valid_lens = torch.repeat_interleave(
                valid_lens,repeats=self.num_heads,dim=0

          )
        # output的形狀(batch_size*num_heads，查詢或‘鍵-值’對的個數，num_hiddens/num_heads)
        output = self.attention(queries,keys,values,valid_lens)

        output_concat = transpose_output(output,self.num_heads)
        return self.W_o(output_concat)
    


def transpose_qkv(X,num_heads):
    '''為了多頭注意力的並行計算而變換形狀'''
    # 輸入X的形狀(batch_size,查詢或‘鍵-值’對的個數，num_hiddens)
    # 輸出X的形狀(batch_size,查詢或‘鍵-值’對的個數，num_heads,num_hiddens/num_heads)
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    
    # 輸出X的形狀(batch_size,num_head,查詢或‘鍵-值’對的個數，num_hiddens/num_heads)
    X = X.permute(0,2,1,3)

    # 最終輸出的形狀(batch_size*num_hiddens,查詢或‘鍵-值’對的個數，num_hiddens/num_heads)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X,num_heads):
    '''逆轉tranpose_qkv函數操作'''
    X= X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    return X.reshape(X.shape[0], X.shape[1], -1)

if __name__=='__main__':
    num_hiddens = 100 
    num_heads = 5 
    attention = MultiHeadAttention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,.5)
    attention.eval()
    batch_size = 2 
    num_queries = 4 
    num_kvparis = 6 
    valid_lens = torch.tensor([3,2])
    X = torch.ones((batch_size,num_queries,num_hiddens))
    Y = torch.ones((batch_size,num_kvparis,num_hiddens))
    print(attention(X,Y,Y,valid_lens).shape)

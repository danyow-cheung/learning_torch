import torch 
from torch import nn 
from attention_mask_softmax import masked_softmax
from d2l import torch as d2l 
import math 

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs) -> None:
        super(DotProductAttention,self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    #   queries shape (batch_size,查询的个数,d)
    #   keys shape (batch_size,‘键-值’对个数,d)
    #   values shape (batch_size,)或者(batch_size,查询个数)
    #   valid_lens shape(batch_size,)或者(batch_size,查询的个数)
    def forward(self,queries,keys,values,valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b = True 为了交换key的最后两个维度
        scores = torch.bmm(queries,keys.transpose(1,2))//math.sqrt(d)
        self.attention_weights = masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

if __name__=='__main__':
    queries = torch.normal(0,1,(2,1,2))
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    
    attention(queries,keys,values,valid_lens)
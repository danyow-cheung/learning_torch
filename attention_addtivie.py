import torch 
from torch import nn 
from attention_mask_softmax import masked_softmax
from d2l import torch as d2l 

class AdditiveAttention(nn.Module):
    '''加性注意力'''
    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs) -> None:
        super(AdditiveAttention,self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=False)
        self.w_v = nn.Linear(num_hiddens,1,bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,queries,keys,values,valid_lens):
        queries,keys = self.W_q(queries),self.W_k(keys)
        # 在维度扩展后
        # queries的形状(batch_size,查询个数,1,num_hidden)
        # key的形状(batch_size,1,'键-值'对个数,num_hidden)
        # 使用广播方式求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # self.w_v只有一个输出，因此从形状中移除最后那个维度
        # scores的形状(batch_size,查询个数,'键-值'对个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores,valid_lens)
        # values的形状(batch_size,‘键-值’个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights),values)



if __name__=='__main__':

    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = AdditiveAttention(key_size=2,query_size=20,num_hiddens=8,dropout=0.1)
    attention.eval()
    attention(queries,keys,values,valid_lens)

    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')




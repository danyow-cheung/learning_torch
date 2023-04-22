import torch
import math 
from torch import nn 
from d2l import torch as d2l 

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens,dropout,max_len = 1000) -> None:
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 創建一個足夠長的p
        self.P = torch.zeros((1,max_len,num_hiddens))

        X = torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32)/num_hiddens)

        self.P[:,:,0::2]=torch.sin(X)
        self.P[:,:,1::2]=torch.cos(X)
    
    def forward(self,X):
        # X = X + self.P[:, :X.shape[1], :].to(X.device)
        X = X + self.P[:, :X.shape[1], :]        
        return self.dropout(X)
    

if __name__=='__main__':
    encoding_dim = 32 
    num_steps = 60 
    pos_encoding = PositionalEncoding(encoding_dim,0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1,num_steps,encoding_dim)))
    P = pos_encoding.P[:,:X.shape[1],:]
    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
                      figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
    

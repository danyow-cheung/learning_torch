import torch
from torch import nn 
from d2l import torch as d2l 

batch_size = 32 
num_steps = 35 

train_iter,vocab = d2l.load_data_time_machine(batch_size,num_steps)

'''1. 初始化模型参数'''
def get_lstm_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01 
    
    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device)) 

    # 输入门参数
    W_xi,W_hi,b_i = three()
    # 遗忘门参数 
    W_xf,W_hf,b_f = three()
    # 输出门参数 
    W_xo,W_ho,b_o = three()
    # 候选记忆元参数
    W_xc,W_hc,b_c = three()
    
    # 输出层参数
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device=device)

    # 附加梯度 
    params = [
        W_xi,W_hi,b_i,
        W_xf,W_hf,b_f,
        W_xo,W_ho,b_o,
        W_xc,W_hc,b_c,
        W_hq,b_q
    ]
    for param in params:
        param.requires_grad_(True)
    return params 

'''2. 定义模型'''
def init_lstm_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),
            torch.zeros((batch_size,num_hiddens),device=device))


def lstm(inputs,state,params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,W_hq, b_q] = params
    (H,C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X@W_xi)+(H@W_hi)+b_i)
        F = torch.sigmoid((X@W_xf)+(H@W_hf)+b_f)
        O = torch.sigmoid((X@W_xo)+(H@W_ho)+b_o)

        C_tilda =torch.tanh((X@W_xc)+(H@W_hc)+b_c)
        C = F*C + I*C_tilda
        H = O*torch.tanh(C)
        Y = (H@W_hq)+b_q 
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,C)

'''3. 训练和预测'''
vocab_size = len(vocab)
num_hiddens = 256 
device = d2l.try_gpu(0)
num_epochs = 500 
lr =  1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
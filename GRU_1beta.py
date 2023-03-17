import torch
from torch import nn 
from d2l import torch as d2l 
'''
GRU_1.py的简洁实现
'''

batch_size = 32 
num_steps = 35 

train_iter,vocab = d2l.load_data_time_machine(batch_size,num_steps)

'''1. 初始化模型参数'''
def get_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        # Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
        # 返回一个由均值为0，方差为1的正态分布(也称为标准正态分布)中的随机数填充的张量。
        return torch.randn(size=shape,device=device)*0.01 
    
    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device)) 
    
    W_xz,W_hz,b_z = three() # 更新⻔参数
    W_xr,W_hr,b_r = three() # 重置⻔参数
    W_xh,W_hh,b_h = three()# 候选隐状态参数

    # 输出层参数
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device=device)
    # 附加梯度
    params = [
        W_xz,W_hz,b_z,
        W_xr,W_hr,b_r,
        W_xh,W_hh,b_h,
        W_hq,b_q
    ]
    for param in params:
        # Change if autograd should record operations on this tensor: sets this tensor’s requires_grad attribute in-place. Returns this tensor.
        # 更改autograd是否应该记录这个张量上的操作:设置这个张量的requires_grad属性。返回这个张量。
        param.requires_grad_(True)

    return params


'''2.定义模型'''
def init_gru_state(batch_size,num_hiddens,device):
    # 用零填充张量形状
    return (torch.zeros((batch_size,num_hiddens),device=device),)

'''3. 定义门控循环单元模型'''
def gru(inputs,state,params):
    W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q = params
    H, = state 
    outputs = []
    for X in inputs:
        # @为矩阵乘法，激活函数为sigmoid
        # torch.sigmoid 返回张量
        
        Z = torch.sigmoid((X @ W_xz)+(H @ W_hz)+b_z)
        R = torch.sigmoid((X@W_xr)+(H@W_hr)+b_r)
        H_tilda = torch.tanh((X@W_xh)+((R*H)@W_hh)+b_h)
        
        H = Z*H + (1-Z)*H_tilda
        Y = H@W_hq + b_q 
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,)

''' 不同之处:直接实例化'''
vocab_size = len(vocab)
num_hiddens = 256 
device = d2l.try_gpu(0)
num_epochs = 500 
lr = 1
num_inputs = vocab_size
gru_layers = nn.GRU(num_inputs,num_hiddens)
model = d2l.RNNModel(gru_layers,len(vocab))
model = model.to(device)
d2l.train_ch8(model,train_iter,vocab,lr,num_epochs,device)


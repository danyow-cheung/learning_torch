from optimize_target import f,g

import torch

f = lambda x:0.5 *x **2# 凸函数
g = lambda x:torch.cos(np.pi*x)  # 非凸函数
h = lambda x:torch.exp(0.5*x) # 凸函数

x,segment = torch.arange(-2,2,0.01),torch.tensor([-1.5,1])


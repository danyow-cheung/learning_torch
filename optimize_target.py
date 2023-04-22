import numpy as np 
import torch 
from d2l import torch as d2l 
import matplotlib.pyplot as plt 

def f(x):
    # 风险函数
    return x*torch.cos(np.pi*x)

def g(x):
    # 经验风险函数
    return f(x)+0.2*torch.cos(5*np.pi*x)

def annotate(text,xy,xytext):
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))
    # plt.gca()

x = torch.arange(0.5,1.5,0.01)
d2l.set_figsize((4.5,2.5))
d2l.plot(x,[f(x),g(x)],'x','risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
   

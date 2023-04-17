'''
注意力的可视化
2023.4.17 danyow
'''
import torch 
from d2l import torch as d2l 

# 可视化注意力权重，需要定义一个show_heatmaps函数，其输入matrices的形状是(要显示的行数，要显示的列数，查询的数目，键的数目)
def show_heatmaps(matrices,xlabel,ylabel,titles=None,figsize=(2.5,2.5),cmap='Reds'):
    '''显示矩阵热图'''
    d2l.use_svg_display()
    num_rows,num_cols = matrices.shape[0],matrices.shape[1]

    fig,axes = d2l.plt.subplots(
        num_rows,num_cols,figsize=figsize,sharex=True,sharey=True,squeeze=False
    
    )

    for i ,(row_axes,row_matrics) in enumerate(zip(axes,matrices)):
        for j,(ax,matrix) in enumerate(zip(row_axes,row_matrics)):
            pcm = ax.imshow(matrix.detach().numpy(),cmap=cmap)
            if i==num_rows-1:
                ax.set_xlabel(xlabel)
            if j==0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm,ax=axes,shrink=0.6)
    

# 测试
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
print(attention_weights.size())

show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
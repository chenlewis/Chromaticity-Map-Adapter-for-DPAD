import torch
import torch.nn as nn

# 创建线性层
linear_layer = nn.Linear(3136, 768)

# 计算线性层的参数数量
total_params = sum(p.numel() for p in linear_layer.parameters())
print("线性层的参数数量:", total_params)

"""
PyTorch 自动求导机制
训练神经网络时最常用的算法就是反度传播算法。
该算法会根据损失函数Loss对不同维度的参数求偏导得到梯度，
然后基于该梯度更新模型权重（w权重项和b偏置项）。
之前在机器学习的模块我们只能通过手动计算的方法来实现求偏导，现在我们可以使用PyTorch的自动求导机制来实现。

1. tensor.backward方法 ：用于计算张量的梯度。
2. tensor.grad属性 ：用于存储张量的梯度。

注意：
1. pyTorch只支持对标量张量进行求偏导，不能对向量张量进行求偏导。
2. pyTorch只支持对对矩阵张量进行求偏导。
"""

# 1. 导入torch库
import torch

# 2. 设置标量张量w
w = torch.tensor(10, requires_grad=True,dtype=torch.float32)
print(f'权重标量w为:{w}')

# 3. 设置损失函数Loss
Loss = 2 * w ** 2
print(f'损失函数Loss为:{Loss}')

# 4. 对损失函数Loss求偏导 计算得到w的梯度 记录在w的grad属性中
Loss.backward()
print(f'w的梯度为:{w.grad}')

# 5. 更新权重w(w1 = w0-学习率*梯度)
w.data = w.data - 0.01 * w.grad
print(f'更新后的权重w为:{w}')

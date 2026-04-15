"""
基于pytorch的自动求导机制结合梯度下降法求最优权重参数
"""
# 1. 导入torch库
import torch

# 2. 定义权重w初始值，设置值为标量，并且开启梯度计算功能
w = torch.tensor(10, requires_grad=True,dtype=torch.float32)
print(f'权重标量w为:{w}')

# 2. 设置损失函数Loss
Loss = w ** 2 + 20
print(f'损失函数Loss为:{Loss}')

# 3. 基于梯度下降法更新权重w(w1 = w0-学习率*梯度)
for i in range(1,101):
    # 3.1 正向传播 因为当w.data更新之后，w也会更新，因此这里的损失值Loss也需要更新
    Loss = w ** 2 + 20

    # 3.2 梯度清零（为什么需要清零？）
    if w.grad is not None:
        w.grad.zero_()
    
    # 3.3 反向传播（自动对损失函数求在w分量上的偏导 记录在w的grad属性中）
    Loss.backward()

    # 3.4 更新权重w(w1 = w0-学习率*梯度)
    w.data = w.data - 0.01 * w.grad
    print(f'更新后的权重w为:{w}',f'当前损失函数Loss为:{Loss}')

print(f'最终的权重w为:{w}',f'最终的损失函数Loss为:{Loss}')
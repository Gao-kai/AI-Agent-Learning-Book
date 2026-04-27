"""
自动微分真实应用场景
1. 先前向传播计算出预测值
2. 计算损失函数Loss
3. 基于损失函数的预测值z减去真实值y以及自动微分系统计算梯度grad
4. 反向传播：结合权重更新公式 w新 = w旧 - 学习率 * 梯度grad
"""

import torch


# 1. 创建一个3行5列的特征列x的张量,假设值为全1
x = torch.ones(3,5)

# 2. 创建一个5行4列的权重w的张量，值为正态随机分布
w = torch.randn(5,4,requires_grad=True)

# 3. 创建一个1行4列的偏置项b的张量，值为正态随机分布
b = torch.randn(1,4,requires_grad=True)

# 4. 创建一个3行4列的真实值y的张量，假设值为全0
y = torch.zeros(3,4)

print(f"特征类x的张量为：{x}")
print(f"真实值y的张量为：{y}")
print(f"偏置项b的张量为：{b}")
print(f"权重w的张量为：{w}")


# 4. 计算预测值z = w @ x + b 会发生广播机制，将b扩展为3行4列
z = torch.matmul(x,w) + b;
print(f"预测值z的张量为：{z}")


"""
5. 创建一个均方误差损失函数的实例
- 接收两个张量作为输入 ：预测值和真实值
- 计算它们之间的均方误差
- 返回一个标量张量 ，表示计算得到的损失值
"""
criterion = torch.nn.MSELoss()
Loss = criterion(z,y)
print(f"损失函数Loss的张量为：{Loss}")


# 7. 循环更新权重w和偏置项b
learning_rate = 0.01
for epoch in range(10):
    # 前向传播计算预测值z
    z = torch.matmul(x,w) + b;
    # 计算损失函数Loss
    Loss = criterion(z,y)
    # 反向传播计算梯度grad
    Loss.backward()
    # 更新权重w和偏置项b
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad
    # 清空梯度grad 避免累加计算
    w.grad.zero_()
    b.grad.zero_()
    print(f"第{epoch}轮w为：{w.data}")
    print(f"第{epoch}轮b为：{b.data}")
    print(f"第{epoch}轮损失值为：{Loss.item()}")
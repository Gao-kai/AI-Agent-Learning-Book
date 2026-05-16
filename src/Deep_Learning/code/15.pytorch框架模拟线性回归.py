import torch
from torch.utils.data import TensorDataset  # 张量数据集模块
from torch.utils.data import DataLoader  # 数据加载器模块
from torch import nn  # 神经网络模块 内含MSELoss损失函数和Linear层
from torch import optim  # 优化器函数
from sklearn.datasets import make_regression  # 创建线性回归数据集
import matplotlib.pyplot as plt # 绘图

plt.rcParams['font.sans-serif'] = ['Consolas']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题


"""
基于sklearn的make_regression函数创建线性回归数据集合

可以快速的生成线性回归数据集 函数返回值为
1. X（特征矩阵）类型为numpy对象
2. y（目标值向量）
3. coef（生成数据时使用的真实系数向量）
"""
def create_dataset():
  
    
    X, y, coef = make_regression(
        n_samples=100, # 样本数量
        n_features=1,  # X特征数量
        noise=10,      # 噪声数量
        bias=30,       # 截距
        coef=True,    # 是否返回系数向量 默认False 为True 函数会返回三个值： X （特征矩阵）、 y （目标值向量）和 coef （生成数据时使用的真实系数向量）
        random_state=100, # 随机种子
    )

    print(f'特征矩阵X: {X},X的类型为{type(X)}') # <class 'numpy.ndarray'>
    print(f'目标值向量y: {y},y的类型为{type(y)}') # <class 'numpy.ndarray'>
    print(f'真实系数向量coef: {coef},coef的类型为{type(coef)}')

    # 将numpy数组转换为tensor张量 才可以进一步成为TensorDataset数据集 -> 最后经过数据加载器加载数据集
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y, coef


"""
模型训练
"""
def train_model():
    # 1. 将训练数据集从tensor张量=>TensorDataset数据集=>DataLoader数据加载器
    dateset = TensorDataset(X,y)

    # 2. 创建数据加载器对象
    # dateset: 数据集对象
    # batch_size: 每个批次的样本数量
    # shuffle: 是否随机打乱数据集
    dataloader = DataLoader(dateset, batch_size=16, shuffle=True)
    
    # 3. 创建初始线性回归模型
    # in_features: 输入特征数量 本案例中只有一个特征列 因此为1
    # out_features: 输出特征数量 本案例中只有一个目标值列 因此为1
    model = nn.Linear(in_features=1, out_features=1)


    # 4. 定义损失函数
    # MSELoss（Mean Squared Error Loss）: 均方误差损失函数
    criterion = nn.MSELoss()

    # 5. 定义优化器
    # SGD（Stochastic Gradient Descent）: 随机梯度下降优化器
    # 参数model.parameters(): 模型参数迭代器
    # 参数lr: 学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 6. 具体训练过程

if __name__ == '__main__':
    X, y, coef = create_dataset()
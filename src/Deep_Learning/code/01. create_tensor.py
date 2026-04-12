"""
创建张量的方式
1. torch.tensor 根据指定数据创建
2. torch.Tensor 根据形状创建张量 或者 也可以根据数据创建张量
3. torch.IntTensor 根据整数数据创建张量
4. torch.FloatTensor 根据浮点数数据创建张量
5. torch.DoubleTensor 根据双精度浮点数数据创建张量

注意：
1. torch.tensor 创建张量时，可以手动指定数据类型。
    如果未指定数据类型，默认创建的张量类型为 torch.float32。
    如果数据和指定的数据类型不匹配会自动转换为指定的数据类型。
2. torch.Tensor 创建张量时，不可以手动指定数据类型

"""

import torch
import numpy as np

# 通过torch.tensor创建张量
# torch.tensor支持根据数据创建张量
# torch.tensor支持创建张量的同时指定数据类型
# torch.tensor不支持指定形状创建张量
def runTest01():
    # 通过标量创建
    t1 = torch.tensor(10)
    print(f't1:{t1}')
    print(f't1类型:{type(t1)}')
    print('='* 30)

    # 通过数组量创建
    t2 = torch.tensor([1,2,3])
    print(f't2:{t2}')
    print(f't2类型:{type(t2)}')
    print('='* 30)

    # 通过二维数组创建
    t3 = torch.tensor([[1,2,3],[4,5,6]])
    print(f't3:{t3}')
    print(f't3类型:{type(t3)}')
    print('='* 30)

    # 通过numpy数组创建
    data = np.random.randint(0,10,(3,2))
    t4 = torch.tensor(data,dtype=torch.int32)
    print(f't4:{t4}')
    print(f't4类型:{type(t4)}')
    print('='* 30)

# 通过torch.Tensor创建张量
# torch.Tensor支持根据数据创建张量
# torch.Tensor不支持创建张量的同时指定数据类型
# torch.Tensor支持指定形状创建张量
def runTest02():
    # 通过标量创建
    t1 = torch.Tensor(10)
    print(f't1:{t1}')
    print(f't1类型:{type(t1)}')
    print('='* 30)

    # 通过数组量创建
    t2 = torch.Tensor([1,2,3])
    print(f't2:{t2}')
    print(f't2类型:{type(t2)}')
    print('='* 30)

    # 通过二维数组创建
    t3 = torch.Tensor([[1,2,3],[4,5,6]])
    print(f't3:{t3}')
    print(f't3类型:{type(t3)}')
    print('='* 30)

    # 通过numpy数组创建
    data = np.random.randint(0,10,(3,2))
    t4 = torch.Tensor(data)
    print(f't4:{t4}')
    print(f't4类型:{type(t4)}')
    print('='* 30)

    # 通过指定形状创建张量
    t5 = torch.Tensor(3,5)
    print(f't5:{t5}')
    print(f't5类型:{type(t5)}')
    print('='* 30)


if __name__ == '__main__':
    # runTest01()
    runTest02()

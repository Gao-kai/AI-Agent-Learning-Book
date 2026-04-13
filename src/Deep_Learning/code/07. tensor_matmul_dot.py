"""
（1）张量的点乘运算
规则
- 1. 点乘只能对相同形状的张量进行操作
- 2. 点乘之后元素不相加，这是和dot的区别

方法
- 1. 点乘使用t1.mul(t2)方法
- 2. 点乘也可以直接通过运算符*实现（⭐️）


（2）张量的矩阵乘法运算
规则
- 1. 第一个张量的列数必须等于第二个张量的行数（前列=后行才可乘）
- 2. 如果满足条件1，那么矩阵相乘的结果为一个Size为前行数*后列数的张量

方法
- 1. 二阶矩阵乘法使用torch.mm(a,b)方法
- 2. 矩阵乘法也可以使用torch.matmul(a,b)方法
- 3. 矩阵乘法也可以直接通过运算符 a @ b实现
- 4. 矩阵乘法也可以使用a.matmul(b)方法

（3）一维张量的矩阵乘法运算
- 使用a.dot(b)方法 点积, 将ab两个张量的对应元素相乘后相加，得到一个标量
- 一般用于降维成标量（向量），或进行张量收缩

"""

import torch


def demo_tensor_mul():
    """
    torch.Tensor.mul(other) / t1 * t2
    - other: 与当前张量相乘的张量或标量
    - 返回: 逐元素相乘的结果张量（原张量不变）
    """
    print("\n========== torch.Tensor.mul / * ==========")
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([4.0, 5.0, 6.0])
    result = t1.mul(t2)
    print(f"t1 = {t1.tolist()}")
    print(f"t2 = {t2.tolist()}")
    print(f"t1.mul(t2) = {result.tolist()}")
    print(f"t1 * t2 = {(t1 * t2).tolist()}")



def demo_torch_mm():
    """
    torch.mm(mat1, mat2)
    - mat1: 第一个矩阵，shape (m, n)
    - mat2: 第二个矩阵，shape (n, p)
    - 返回: 矩阵乘法结果，shape (m, p)，仅支持2D张量
    """
    print("\n========== torch.mm ==========")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    result = torch.mm(A, B)
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"torch.mm(A, B) = \n{result}")


def demo_torch_matmul():
    """
    torch.matmul(input, other)
    - input: 第一个张量
    - other: 第二个张量
    - 返回:
        - 1D + 1D: 点积（标量）
        - 2D + 2D: 矩阵乘法
        - N维 + N维: 批量矩阵乘法（支持广播）
    """
    print("\n========== torch.matmul / @ ==========")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    result = torch.matmul(A, B)
    print(f"torch.matmul(A, B) = \n{result}")
    print(f"A @ B = \n{A @ B}")


def demo_tensor_matmul():
    """
    torch.Tensor.matmul(other) / a.matmul(b)
    - other: 第二个张量
    - 返回: 与torch.matmul相同，根据输入维度执行不同乘法
    """
    print("\n========== torch.Tensor.matmul ==========")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    result = A.matmul(B)
    print(f"A.matmul(B) = \n{result}")


def demo_torch_dot():
    """
    torch.dot(a, b)
    - a: 第一个一维张量
    - b: 第二个一维张量
    - 返回: 点积结果（标量），将对应元素相乘后求和
    """
    print("\n========== torch.dot ==========")
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([4.0, 5.0, 6.0])
    result = torch.dot(v1, v2)
    print(f"v1 = {v1.tolist()}")
    print(f"v2 = {v2.tolist()}")
    print(f"torch.dot(v1, v2) = {result.item()}")


if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch 张量乘法运算演示")
    print("=" * 50)

    demo_tensor_mul()
    demo_torch_mm()
    demo_torch_matmul()
    demo_torch_dot()

    print("\n" + "=" * 50)
    print("演示完成")
    print("=" * 50)

"""
张量常见计算函数

（1）以下四个函数都有dim参数 0表示按列计算，1表示按行计算 dim的意思是按哪个维度计算，默认值为None
1.均值mean
2.求和sum
3.最大值max
4.最小值min


（2）以下函数没有dim参数
1.标准差std
2.方差var
3.平方根sqrt 对每一个张量的元素进行开平方计算
4.指数计算exp 对每一个张量的元素进行指数计算 比如假设t = [1,3,5] e的1次方 3次方 5次方
5.对数计算log/log2/log10 对每一个张量的元素进行对数计算 比如假设t = [1,3,5] log1 log3 log5
6.幂运算 pow 求每一个张量的指数值 比如平方 立方 也可以用符号** 来表示
"""

import torch


def demo_tensor_mean():
    """
    torch.Tensor.mean(dim=None)
    - dim: 指定维度，不指定则计算所有元素的均值
    - 返回: 均值张量
    """
    print("\n========== torch.mean ==========")
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = t.mean()
    print(f"t = {t.tolist()}")
    print(f"t.mean() = {result.item()}")

    t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"\nt2 = \n{t2}")
    print(f"t2.mean(dim=0) = {t2.mean(dim=0).tolist()}")
    print(f"t2.mean(dim=1) = {t2.mean(dim=1).tolist()}")


def demo_tensor_sum():
    """
    torch.Tensor.sum(dim=None)
    - dim: 指定维度，不指定则计算所有元素的和
    - 返回: 求和结果张量
    """
    print("\n========== torch.sum ==========")
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = t.sum()
    print(f"t = {t.tolist()}")
    print(f"t.sum() = {result.item()}")

    t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"\nt2 = \n{t2}")
    print(f"t2.sum(dim=0) = {t2.sum(dim=0).tolist()}")
    print(f"t2.sum(dim=1) = {t2.sum(dim=1).tolist()}")


def demo_tensor_max():
    """
    torch.Tensor.max(dim=None)
    - dim: 指定维度，不指定则返回所有元素的最大值
    - 返回: 最大值张量，若指定dim则同时返回最大值和索引
    """
    print("\n========== torch.max ==========")
    t = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    result = t.max()
    print(f"t = {t.tolist()}")
    print(f"t.max() = {result.item()}")

    t2 = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    print(f"\nt2 = \n{t2}")
    print(f"t2.max(dim=0) = {t2.max(dim=0)}")
    print(f"t2.max(dim=1) = {t2.max(dim=1)}")


def demo_tensor_min():
    """
    torch.Tensor.min(dim=None)
    - dim: 指定维度，不指定则返回所有元素的最小值
    - 返回: 最小值张量，若指定dim则同时返回最小值和索引
    """
    print("\n========== torch.min ==========")
    t = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    result = t.min()
    print(f"t = {t.tolist()}")
    print(f"t.min() = {result.item()}")

    t2 = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    print(f"\nt2 = \n{t2}")
    print(f"t2.min(dim=0) = {t2.min(dim=0)}")
    print(f"t2.min(dim=1) = {t2.min(dim=1)}")


def demo_tensor_std():
    """
    torch.Tensor.std()
    - 无dim参数，计算所有元素的标准差
    - 返回: 标准差张量
    """
    print("\n========== torch.std ==========")
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = t.std()
    print(f"t = {t.tolist()}")
    print(f"t.std() = {result.item()}")


def demo_tensor_var():
    """
    torch.Tensor.var()
    - 无dim参数，计算所有元素的方差
    - 返回: 方差张量
    """
    print("\n========== torch.var ==========")
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = t.var()
    print(f"t = {t.tolist()}")
    print(f"t.var() = {result.item()}")


def demo_tensor_sqrt():
    """
    torch.Tensor.sqrt()
    - 无参数，计算每个元素的平方根
    - 返回: 平方根张量
    """
    print("\n========== torch.sqrt ==========")
    t = torch.tensor([1.0, 4.0, 9.0, 16.0])
    result = t.sqrt()
    print(f"t = {t.tolist()}")
    print(f"t.sqrt() = {result.tolist()}")


def demo_torch_exp():
    """
    torch.exp(tensor)
    - 无dim参数，计算e^x
    - 返回: e^x 张量
    """
    print("\n========== torch.exp ==========")
    t = torch.tensor([0.0, 1.0, 2.0])
    result = torch.exp(t)
    print(f"t = {t.tolist()}")
    print(f"torch.exp(t) = {result.tolist()}")


def demo_torch_log():
    """
    torch.log(tensor)
    - 无dim参数，计算ln(x)
    - 返回: 自然对数 ln(x) 张量
    """
    print("\n========== torch.log ==========")
    t = torch.tensor([1.0, 2.0, 3.0])
    result = torch.log(t)
    print(f"t = {t.tolist()}")
    print(f"torch.log(t) = {result.tolist()}")


if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch 张量计算函数演示")
    print("=" * 50)

    print("\n--- 有dim参数的函数 ---")
    demo_tensor_mean()
    demo_tensor_sum()
    demo_tensor_max()
    demo_tensor_min()

    print("\n--- 无dim参数的函数 ---")
    demo_tensor_std()
    demo_tensor_var()
    demo_tensor_sqrt()
    demo_torch_exp()
    demo_torch_log()

    print("\n" + "=" * 50)
    print("演示完成")
    print("=" * 50)

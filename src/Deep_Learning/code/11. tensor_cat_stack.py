"""
张量的拼接操作

1. cat函数
   - 作用: 按指定维度拼接多个张量，除了拼接维度之外的其他维度必须保持相同
   - 输入参数:
      - tensors: 要拼接的张量列表
      - dim: 拼接的维度 默认值为0
   - 输出: 返回一个新的张量，所有输入张量在指定维度上拼接起来
2. stack函数
   - 作用: 在新维度上堆叠张量，所有维度必须保持相同，会插入新的维度
   - 输入参数:
      - tensors: 要拼接的张量列表
      - dim: 拼接的维度 默认值为0
   - 输出: 返回一个新的张量，所有输入张量在指定维度上拼接起来
"""
import torch


def demo_cat():
    """
    cat 函数
    作用: 按指定维度拼接多个张量
    输入参数:
        - tensors: 要拼接的张量列表
        - dim: 拼接的维度，默认为 0
    输出: 返回一个新的张量，所有输入张量在指定维度上拼接起来
    说明: 除了拼接维度之外，其他维度必须保持相同
    """
    print("【cat: 按指定维度拼接张量】")

    print("\n1. 默认按 dim=0 拼接二维张量:")
    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
    print("张量1:\n", tensor1)
    print("张量2:\n", tensor2)

    result = torch.cat([tensor1, tensor2])
    print("\n使用 cat([tensor1, tensor2], dim=0) 拼接后:")
    print("形状:", result.shape)
    print("结果:\n", result)

    print("\n" + "="*50)
    print("\n2. 按 dim=1 拼接（横向拼接）:")
    tensor3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor4 = torch.tensor([[7, 8], [9, 10]])
    print("张量1:\n", tensor3)
    print("张量2:\n", tensor4)

    result = torch.cat([tensor3, tensor4], dim=1)
    print("\n使用 cat([tensor1, tensor2], dim=1) 拼接后:")
    print("形状:", result.shape)
    print("结果:\n", result)

    print("\n" + "="*50)
    print("\n3. 拼接三维张量:")
    t1 = torch.randint(1, 10, (2, 3, 4))
    t2 = torch.randint(1, 10, (3, 3, 4))
    t3 = torch.randint(1, 10, (1, 3, 4))
    print("张量1形状:", t1.shape)
    print("张量2形状:", t2.shape)
    print("张量3形状:", t3.shape)

    result = torch.cat([t1, t2, t3], dim=0)
    print("\n使用 cat([t1, t2, t3], dim=0) 拼接后:")
    print("形状:", result.shape)
    print("结果:\n", result)

    print("\n" + "="*50)
    print("\n4. 拼接三维张量的其他维度:")
    t4 = torch.randint(1, 10, (2, 3, 4))
    t5 = torch.randint(1, 10, (2, 5, 4))
    print("张量1形状:", t4.shape)
    print("张量2形状:", t5.shape)

    result = torch.cat([t4, t5], dim=1)
    print("\n使用 cat([t4, t5], dim=1) 拼接后:")
    print("形状:", result.shape)


def demo_stack():
    """
    stack 函数
    作用: 按指定维度拼接多个张量，会在指定位置插入一个新维度
    输入参数:
        - tensors: 要拼接的张量列表
        - dim: 拼接的维度，默认为 0
    输出: 返回一个新的张量，所有输入张量在指定维度上拼接起来
    说明: 所有输入张量必须有完全相同的形状，会在指定维度前插入一个维度
    """
    print("【stack: 在新维度上堆叠张量】")

    print("\n1. 默认按 dim=0 堆叠一维张量:")
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])
    tensor3 = torch.tensor([7, 8, 9])
    print("张量1:", tensor1)
    print("张量2:", tensor2)
    print("张量3:", tensor3)

    result = torch.stack([tensor1, tensor2, tensor3])
    print("\n使用 stack([t1, t2, t3], dim=0) 堆叠后:")
    print("形状:", result.shape)
    print("结果:\n", result)

    print("\n" + "="*50)
    print("\n2. 按 dim=1 堆叠:")
    result = torch.stack([tensor1, tensor2, tensor3], dim=1)
    print("使用 stack([t1, t2, t3], dim=1) 堆叠后:")
    print("形状:", result.shape)
    print("结果:\n", result)

    print("\n" + "="*50)
    print("\n3. 堆叠二维张量:")
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])
    print("张量1:\n", t1)
    print("张量2:\n", t2)

    result = torch.stack([t1, t2], dim=0)
    print("\n使用 stack([t1, t2], dim=0) 堆叠后:")
    print("形状:", result.shape)
    print("结果:\n", result)

    result = torch.stack([t1, t2], dim=2)
    print("\n使用 stack([t1, t2], dim=2) 堆叠后:")
    print("形状:", result.shape)
    print("结果:\n", result)

    print("\n" + "="*50)
    print("\n4. cat vs stack 的区别:")
    t_a = torch.tensor([1, 2, 3])
    t_b = torch.tensor([4, 5, 6])

    cat_result = torch.cat([t_a, t_b], dim=0)
    stack_result = torch.stack([t_a, t_b], dim=0)

    print("张量a:", t_a)
    print("张量b:", t_b)
    print("\n原始形状: a={}, b={}".format(t_a.shape, t_b.shape))

    print("\ncat([a, b], dim=0) 结果形状:", cat_result.shape)
    print("cat 结果:", cat_result)

    print("\nstack([a, b], dim=0) 结果形状:", stack_result.shape)
    print("stack 结果:\n", stack_result)

    print("\n【区别说明】")
    print("- cat: 拼接后总元素数 = 6，结果形状 (6,)")
    print("- stack: 堆叠后会插入新维度，结果形状 (2, 3)")


if __name__ == '__main__':
    demo_cat()
    print("\n" + "="*60)
    demo_stack()
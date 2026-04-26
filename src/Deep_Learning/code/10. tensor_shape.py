"""
张量的形状操作API

1. reshape 转换张量形状
2. squeeze 移除张量中所有尺寸为1的维度
3. unsqueeze 在张量的指定位置插入尺寸为1的维度
4. transpose 交换张量的两个维度 一次只能交换两个维度
5. permute 重新排列张量的维度 一次可以交换多个维度的顺序
6. view 重新调整张量的形状 只能用于修改连续的张量 比如一旦一个张量已经通过transpose或permute操作 就不能使用view方法
7. contiguous 将张量转换为连续存储的张量
8. is_contiguous 检查张量是否为连续存储的张量
"""
import torch

"""
reshape函数
1. tensor.shape 方法 返回张量的形状 返回值是一个元组 二维张量的形状为(行数,列数)
2. tensor.reshape 方法 
- 重新调整张量的形状 返回值是一个新的张量 必须保证转换前后元素数量一致
- 不会改变张量内容的情况下 修改张量的形状

"""

def demo_reshape():
    tensor = torch.randint(1,10,(2,3))
    print(tensor)
    print(tensor.shape)
    print(f"行数为{tensor.shape[0]}，列数为{tensor.shape[1]}")

    # 转换为3行2列
    print(tensor.reshape(3,2))
    # 转换为1行6列
    print(tensor.reshape(1,6))
    # 转换为6行1列
    print(tensor.reshape(6,1))

def demo_squeeze():
    """
    squeeze 函数
    作用: 移除张量中所有尺寸为1的维度
    输入参数: 无（也可以通过 dim 参数指定要移除的特定维度）
    输出: 返回一个新的张量，指定的尺寸为1的维度被移除
    """
    tensor = torch.randint(1, 10, (1, 3, 2, 4, 1))
    print("原始张量形状:", tensor.shape)
    print("原始张量:\n", tensor)

    squeezed = tensor.squeeze()
    print("\n使用 squeeze() 移除所有尺寸为1的维度后:")
    print("形状:", squeezed.shape)
    print("张量:\n", squeezed)

    tensor2 = torch.randint(1, 10, (1, 3, 1, 4))
    print("\n" + "="*50)
    print("原始张量形状:", tensor2.shape)
    squeezed_dim = tensor2.squeeze(dim=2)
    print("使用 squeeze(dim=2) 移除指定维度后:")
    print("形状:", squeezed_dim.shape)
    print("张量:\n", squeezed_dim)


def demo_unsqueeze():
    """
    unsqueeze 函数
    作用: 在张量的指定位置插入尺寸为1的维度
    输入参数:
        - dim: 要插入尺寸为1的维度的位置索引
    输出: 返回一个新的张量，在指定位置添加了尺寸为1的维度
    """
    tensor = torch.randint(1, 10, (2, 3))
    print("原始张量形状:", tensor.shape)
    print("原始张量:\n", tensor)

    unsqueezed_dim0 = tensor.unsqueeze(dim=0)
    print("\n使用 unsqueeze(dim=0) 在维度0前插入1后:")
    print("形状:", unsqueezed_dim0.shape)
    print("张量:\n", unsqueezed_dim0)

    unsqueezed_dim1 = tensor.unsqueeze(dim=1)
    print("\n使用 unsqueeze(dim=1) 在维度1前插入1后:")
    print("形状:", unsqueezed_dim1.shape)
    print("张量:\n", unsqueezed_dim1)

    unsqueezed_dim2 = tensor.unsqueeze(dim=2)
    print("\n使用 unsqueeze(dim=2) 在维度2前插入1后:")
    print("形状:", unsqueezed_dim2.shape)
    print("张量:\n", unsqueezed_dim2)


def demo_transpose():
    """
    transpose 函数
    作用: 交换张量的两个维度（一次只能交换两个维度）
    输入参数:
        - dim0: 要交换的第一个维度索引
        - dim1: 要交换的第二个维度索引
    输出: 返回一个新的张量，两个指定维度被交换位置
    """
    tensor = torch.randint(1, 10, (2, 3, 4))
    print("原始张量形状:", tensor.shape)
    print("原始张量:\n", tensor)

    transposed_0_1 = tensor.transpose(0, 1)
    print("\n使用 transpose(0, 1) 交换维度0和维度1后:")
    print("形状:", transposed_0_1.shape)
    print("张量:\n", transposed_0_1)

    tensor2 = torch.randint(1, 10, (2, 3))
    print("\n" + "="*50)
    print("二维张量原始形状:", tensor2.shape)
    print("原始张量:\n", tensor2)
    transposed_2d = tensor2.t()
    print("\n使用 t() 转置二维张量后（相当于 transpose(0, 1）:")
    print("形状:", transposed_2d.shape)
    print("张量:\n", transposed_2d)


def demo_permute():
    """
    permute 函数
    作用: 重新排列张量的维度（一次可以交换多个维度的顺序）
    输入参数:
        - dims: 一个元组，指定各维度的新顺序（每个元素是对应维度的新位置索引）
    输出: 返回一个新的张量，维度按指定顺序重新排列
    """
    tensor = torch.randint(1, 10, (2, 3, 4, 5))
    print("原始张量形状:", tensor.shape)
    print("原始张量 (部分展示):\n", tensor)

    permuted = tensor.permute(2, 1, 0, 3)
    print("\n使用 permute(2, 1, 0, 3) 重新排列维度后:")
    print("新维度顺序: 原维度2 -> 位置0, 原维度1 -> 位置1, 原维度0 -> 位置2, 原维度3 -> 位置3")
    print("形状:", permuted.shape)
    print("张量:\n", permuted)

    tensor2 = torch.randint(1, 10, (2, 3, 4))
    print("\n" + "="*50)
    print("三维张量原始形状:", tensor2.shape)
    print("原始张量:\n", tensor2)
    permuted_3d = tensor2.permute(2, 0, 1)
    print("\n使用 permute(2, 0, 1) 重新排列维度后:")
    print("形状:", permuted_3d.shape)
    print("张量:\n", permuted_3d)

if __name__ == '__main__':
    # demo_reshape()
    print("\n" + "="*50)
    # demo_squeeze()
    print("\n" + "="*50)
    # demo_unsqueeze()
    print("\n" + "="*50)
    # demo_transpose()
    print("\n" + "="*50)
    # demo_permute()
    print("\n" + "="*50)
    demo_view()
    print("\n" + "="*50)
    demo_contiguous()
    print("\n" + "="*50)
    demo_is_contiguous()


def demo_view():
    """
    view 函数
    作用: 重新调整张量的形状，只能用于连续存储的张量
    输入参数:
        - shape: 目标形状（元组），必须保证转换前后元素数量一致
    输出: 返回一个新的张量，形状被重新调整
    注意: 一旦张量通过 transpose 或 permute 操作后，就不能使用 view 方法，
          因为这些操作会创建不连续的张量。此时需要先调用 contiguous() 方法
    """
    print("【连续张量使用 view】")
    tensor = torch.randint(1, 10, (2, 3))
    print("原始张量形状:", tensor.shape)
    print("原始张量:\n", tensor)
    print("is_contiguous:", tensor.is_contiguous())

    view_result = tensor.view(3, 2)
    print("\n使用 view(3, 2) 重新调整形状后:")
    print("形状:", view_result.shape)
    print("张量:\n", view_result)

    print("\n" + "="*50)
    print("【不连续张量不能直接使用 view】")
    tensor2 = torch.randint(1, 10, (2, 3, 4))
    transposed = tensor2.transpose(1, 2)
    print("原始张量形状:", tensor2.shape)
    print("is_contiguous:", tensor2.is_contiguous())
    print("\n使用 transpose(1, 2) 交换维度后:")
    print("新形状:", transposed.shape)
    print("is_contiguous:", transposed.is_contiguous())

    print("\n尝试对不连续张量使用 view 会报错，需要先调用 contiguous()")


def demo_contiguous():
    """
    contiguous 函数
    作用: 将张量转换为连续存储的张量
    输入参数: 无
    输出: 返回一个新的连续张量（如果原张量已经连续，则返回原张量的视图）
    注意: 调用此方法可能会复制数据到新的内存位置
    """
    print("【不连续张量转连续张量】")
    tensor = torch.randint(1, 10, (2, 3, 4))
    print("原始张量形状:", tensor.shape)
    print("原始 is_contiguous:", tensor.is_contiguous())

    transposed = tensor.transpose(1, 2)
    print("\n使用 transpose(1, 2) 后:")
    print("形状:", transposed.shape)
    print("is_contiguous:", transposed.is_contiguous())

    contiguous_tensor = transposed.contiguous()
    print("\n使用 contiguous() 转换为连续张量后:")
    print("形状:", contiguous_tensor.shape)
    print("is_contiguous:", contiguous_tensor.is_contiguous())
    print("张量:\n", contiguous_tensor)


def demo_is_contiguous():
    """
    is_contiguous 函数
    作用: 检查张量是否为连续存储的张量
    输入参数: 无
    输出: 返回一个布尔值，True 表示张量在内存中是连续存储的
    说明: 连续张量在内存中按行优先（C-style）排列，不连续张量通常由 transpose、permute 等操作产生
    """
    print("【连续张量 vs 不连续张量】")
    print("\n连续张量的特点:")
    print("- 张量在内存中是连续存储的")
    print("- 元素按行优先（C-style）排列")
    print("- 可以直接使用 view 方法")
    print("- view、reshape 等操作效率更高")

    tensor = torch.randint(1, 10, (2, 3))
    print("\n原始张量 (连续):")
    print("形状:", tensor.shape)
    print("is_contiguous:", tensor.is_contiguous())

    print("\n" + "="*50)
    print("\n不连续张量的特点:")
    print("- 张量在内存中不是连续存储的")
    print("- 通常由 transpose、permute 等操作产生")
    print("- 不能直接使用 view 方法，需要先调用 contiguous()")
    print("- 底层可能需要数据复制才能重新调整形状")

    tensor2 = torch.randint(1, 10, (2, 3, 4))
    transposed = tensor2.transpose(1, 2)
    print("\n使用 transpose(1, 2) 后的张量 (不连续):")
    print("形状:", transposed.shape)
    print("is_contiguous:", transposed.is_contiguous())

    print("\n" + "="*50)
    print("\n【对比实验：reshape vs view】")
    print("\n对于不连续张量，reshape 和 view 的行为不同:")

    tensor3 = torch.randint(1, 10, (2, 3, 4))
    transposed2 = tensor3.transpose(1, 2)

    print("\n不连续张量:")
    print("形状:", transposed2.shape)
    print("is_contiguous:", transposed2.is_contiguous())

    try:
        reshaped = transposed2.reshape(2, 16)
        print("\n使用 reshape(2, 16) 成功（会自动复制数据）:")
        print("形状:", reshaped.shape)
    except Exception as e:
        print("\nreshape 失败:", e)

    try:
        view_result = transposed2.view(2, 16)
        print("\n使用 view(2, 16) 成功:")
        print("形状:", view_result.shape)
    except Exception as e:
        print("\nview 失败（不连续张量不能直接使用 view）:", e)
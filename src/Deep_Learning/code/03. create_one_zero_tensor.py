"""
创建全0张量
1. torch.zeros 创建全0张量 比如在设置偏置项时使用
2. torch.zeros_like 创建与输入形状相同的全0张量

创建全1张量
1. torch.ones 创建全1张量
2. torch.ones_like 创建与输入形状相同的全1张量

创建指定值张量
1. torch.full 创建指定值张量 
2. torch.full_like 创建与输入形状相同的指定值张量 一般填充值为255 用于创建图像张量
"""

import torch

def demo_torch_zeros():
    """
    torch.zeros(*size, dtype=None, device=None, requires_grad=False)
    - size: 张量形状（如 2, 3 或 (2, 3)）
    - dtype: 数据类型（默认 torch.float32）
    - device: 设备（cpu 或 cuda）
    - requires_grad: 是否需要梯度
    返回: 全0张量
    """
    print("\n========== torch.zeros ==========")
    tensor = torch.zeros(2, 3)
    print(f"torch.zeros(2, 3)")
    print(f"参数: size=(2, 3)")
    print(f"返回值:\n{tensor}")
    print(f"类型: {tensor.dtype}")
    print(f"形状: {tensor.shape}")
    print(f"设备: {tensor.device}")

    tensor_int = torch.zeros(2, 3, dtype=torch.int32)
    print(f"\ntorch.zeros(2, 3, dtype=torch.int32)")
    print(f"返回值:\n{tensor_int}")
    print(f"类型: {tensor_int.dtype}")


def demo_torch_zeros_like():
    """
    torch.zeros_like(input, dtype=None, device=None, requires_grad=False)
    - input: 参考张量，类型必须是Tensor，不可以是普通列表
    - dtype: 数据类型（默认与输入相同）
    - device: 设备（默认与输入相同）
    - requires_grad: 是否需要梯度（默认与输入相同）
    返回: 与输入形状相同的全0张量
    """
    print("\n========== torch.zeros_like ==========")
    ref_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tensor = torch.zeros_like(ref_tensor)
    print(f"参考张量:\n{ref_tensor}")
    print(f"参考张量形状: {ref_tensor.shape}, 类型: {ref_tensor.dtype}")
    print(f"torch.zeros_like(参考张量)")
    print(f"返回值:\n{tensor}")
    print(f"形状: {tensor.shape}, 类型: {tensor.dtype}")


def demo_torch_ones():
    """
    torch.ones(*size, dtype=None, device=None, requires_grad=False)
    - size: 张量形状
    - dtype: 数据类型（默认 torch.float32）
    - device: 设备
    - requires_grad: 是否需要梯度
    返回: 全1张量
    """
    print("\n========== torch.ones ==========")
    tensor = torch.ones(2, 3)
    print(f"torch.ones(2, 3)")
    print(f"参数: size=(2, 3)")
    print(f"返回值:\n{tensor}")
    print(f"类型: {tensor.dtype}")
    print(f"形状: {tensor.shape}")

    tensor_int = torch.ones(2, 3, dtype=torch.int32)
    print(f"\ntorch.ones(2, 3, dtype=torch.int32)")
    print(f"返回值:\n{tensor_int}")
    print(f"类型: {tensor_int.dtype}")


def demo_torch_ones_like():
    """
    torch.ones_like(input, dtype=None, device=None, requires_grad=False)
    - input: 参考张量
    - dtype: 数据类型（默认与输入相同）
    - device: 设备（默认与输入相同）
    - requires_grad: 是否需要梯度
    返回: 与输入形状相同的全1张量
    """
    print("\n========== torch.ones_like ==========")
    ref_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tensor = torch.ones_like(ref_tensor)
    print(f"参考张量:\n{ref_tensor}")
    print(f"参考张量形状: {ref_tensor.shape}, 类型: {ref_tensor.dtype}")
    print(f"torch.ones_like(参考张量)")
    print(f"返回值:\n{tensor}")
    print(f"形状: {tensor.shape}, 类型: {tensor.dtype}")


def demo_torch_full():
    """
    torch.full(size, fill_value, dtype=None, device=None, requires_grad=False)
    - size: 张量形状
    - fill_value: 填充的值
    - dtype: 数据类型（默认根据 fill_value 推断）
    - device: 设备
    - requires_grad: 是否需要梯度
    返回: 指定值的张量
    """
    print("\n========== torch.full ==========")
    tensor = torch.full((2, 3), 255)
    print(f"torch.full((2, 3), 255)")
    print(f"参数: size=(2, 3), fill_value=255")
    print(f"返回值:\n{tensor}")
    print(f"类型: {tensor.dtype}")
    print(f"形状: {tensor.shape}")

    tensor_float = torch.full((2, 3), 3.14)
    print(f"\ntorch.full((2, 3), 3.14)")
    print(f"返回值:\n{tensor_float}")
    print(f"类型: {tensor_float.dtype}")


def demo_torch_full_like():
    """
    torch.full_like(input, fill_value, dtype=None, device=None, requires_grad=False)
    - input: 参考张量
    - fill_value: 填充的值
    - dtype: 数据类型（默认与输入相同）
    - device: 设备（默认与输入相同）
    - requires_grad: 是否需要梯度
    返回: 与输入形状相同的指定值张量
    """
    print("\n========== torch.full_like ==========")
    ref_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tensor = torch.full_like(ref_tensor, 99)
    print(f"参考张量:\n{ref_tensor}")
    print(f"参考张量形状: {ref_tensor.shape}, 类型: {ref_tensor.dtype}")
    print(f"torch.full_like(参考张量, 99)")
    print(f"返回值:\n{tensor}")
    print(f"形状: {tensor.shape}, 类型: {tensor.dtype}")


def demo_special_values():
    """
    特殊值张量
    - torch.zeros_like + requires_grad=True: 创建可训练的张量（神经网络权重初始化常用）
    """
    print("\n========== 特殊用法：创建可训练的张量 ==========")
    ref = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor = torch.zeros_like(ref, requires_grad=True)
    print(f"参考张量:\n{ref}")
    print(f"torch.zeros_like(参考张量, requires_grad=True)")
    print(f"返回值:\n{tensor}")
    print(f"requires_grad: {tensor.requires_grad}")
    print(f"说明: requires_grad=True 表示该张量参与梯度计算，用于神经网络训练")


if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch 全0/全1/指定值张量创建演示")
    print("=" * 50)

    demo_torch_zeros()
    demo_torch_zeros_like()
    demo_torch_ones()
    demo_torch_ones_like()
    demo_torch_full()
    demo_torch_full_like()
    demo_special_values()

    print("\n" + "=" * 50)
    print("演示完成")
    print("=" * 50)

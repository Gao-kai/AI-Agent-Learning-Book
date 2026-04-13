"""
1. 张量 --> numpy中ndarray(✅)
- 使用 tensor.numpy() 可以将张量转换为numpy数组ndarray 共享内存
- 使用 tensor.numpy().copy() 可以将转换后的ndarray复制到新的内存中，避免共享内存问题

2. numpy中ndarray --> 张量

- 使用torch.from_numpy(ndarray) 可以将一个numpy数组ndarray转换为张量tensor 共享内存
- 使用torch.tensor(ndarray) 可以将一个numpy数组ndarray转换为张量tensor     默认不共享内存（用的最多）(✅)

3. 将标量张量（只有一个元素的张量）转换为数字类型(✅)
- 使用tensor.item() 可以将标量张量转换为Python标量（如float、int等）
"""

import torch
import numpy as np


def demo_tensor_to_numpy():
    """
    方法1: tensor.numpy() - 张量转 NumPy 数组
    tensor.numpy()
    - 无参数
    返回: 与张量共享内存的 NumPy 数组
    说明: 转换前后共享内存，修改一方会影响另一方
    """
    print("\n========== tensor.numpy() ==========")
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"原始张量:\n{tensor}")
    print(f"张量类型: {tensor.dtype}")

    numpy_array = tensor.numpy()
    print(f"\ntensor.numpy()")
    print(f"返回值:\n{numpy_array}")
    print(f"NumPy数组类型: {numpy_array.dtype}")

    print(f"\n--- 验证共享内存 ---")
    print(f"tensor 和 numpy_array 是否共享内存: {tensor.storage().data_ptr() == numpy_array.__array_interface__['data'][0]}")

    tensor[0, 0] = 99.0
    print(f"修改 tensor[0,0] = 99 后:")
    print(f"numpy_array[0,0] = {numpy_array[0, 0]}")


def demo_from_numpy():
    """
    方法2: torch.from_numpy() - NumPy 转张量（共享内存）
    torch.from_numpy(ndarray)
    - ndarray: NumPy 数组
    返回: 与 NumPy 共享内存的张量
    说明: 转换前后共享内存，修改一方会影响另一方
    """
    print("\n========== torch.from_numpy() ==========")
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"原始 NumPy 数组:\n{numpy_array}")
    print(f"NumPy 数组类型: {numpy_array.dtype}")

    tensor = torch.from_numpy(numpy_array)
    print(f"\ntorch.from_numpy(numpy_array)")
    print(f"返回值:\n{tensor}")
    print(f"张量类型: {tensor.dtype}")

    print(f"\n--- 验证共享内存 ---")
    print(f"tensor 和 numpy_array 是否共享内存: {tensor.storage().data_ptr() == numpy_array.__array_interface__['data'][0]}")

    numpy_array[0, 0] = 99
    print(f"修改 numpy_array[0,0] = 99 后:")
    print(f"tensor[0,0] = {tensor[0, 0]}")


def demo_torch_tensor():
    """
    方法3: torch.tensor() - NumPy 转张量（不共享内存）
    torch.tensor(ndarray)
    - ndarray: NumPy 数组
    返回: 与 NumPy 不共享内存的新张量（用的最多）
    说明: 默认不共享内存，修改一方不会影响另一方
    """
    print("\n========== torch.tensor() ==========")
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"原始 NumPy 数组:\n{numpy_array}")
    print(f"NumPy 数组类型: {numpy_array.dtype}")

    tensor = torch.tensor(numpy_array)
    print(f"\ntorch.tensor(numpy_array)")
    print(f"返回值:\n{tensor}")
    print(f"张量类型: {tensor.dtype}")

    print(f"\n--- 验证不共享内存 ---")
    print(f"tensor 和 numpy_array 是否共享内存: {tensor.storage().data_ptr() == numpy_array.__array_interface__['data'][0]}")

    tensor[0, 0] = 99.0
    print(f"修改 tensor[0,0] = 99 后:")
    print(f"numpy_array[0,0] = {numpy_array[0, 0]} (未改变)")


def demo_tensor_item():
    """
    方法4: tensor.item() - 标量张量转 Python 标量
    tensor.item()
    - 无参数
    返回: Python 标量值（int, float, bool）
    说明: 只能用于只有一个元素的张量
    """
    print("\n========== tensor.item() ==========")

    tensor_int = torch.tensor(42)
    tensor_float = torch.tensor(3.14159)
    tensor_bool = torch.tensor(True)

    print(f"tensor_int = {tensor_int}, dtype = {tensor_int.dtype}")
    print(f"torch.tensor(42).item() = {tensor_int.item()}, 类型: {type(tensor_int.item())}")

    print(f"\ntensor_float = {tensor_float}, dtype = {tensor_float.dtype}")
    print(f"torch.tensor(3.14159).item() = {tensor_float.item():.5f}, 类型: {type(tensor_float.item())}")

    print(f"\ntensor_bool = {tensor_bool}, dtype = {tensor_bool.dtype}")
    print(f"torch.tensor(True).item() = {tensor_bool.item()}, 类型: {type(tensor_bool.item())}")

    print(f"\n--- 错误示例 ---")
    multi_element_tensor = torch.tensor([1, 2, 3])
    # try:
    #     multi_element_tensor.item()
    # except ValueError as e:
    #     print(f"ValueError: {e}")


def demo_shared_memory_warning():
    """
    共享内存的注意事项
    """
    print("\n========== 共享内存注意事项 ==========")
    print("1. tensor.numpy() 和 torch.from_numpy() 会共享内存")
    print("2. 修改一方的值，另一方也会改变")
    print("3. torch.tensor() 不会共享内存（推荐使用）")
    print("4. tensor.item() 只能用于单元素张量")


def demo_numpy_to_tensor_dtype():
    """
    NumPy 转张量时的类型对应关系
    """
    print("\n========== NumPy -> Tensor 类型对照 ==========")
    print(f"{'NumPy dtype':<20} {'PyTorch dtype':<20}")
    print("-" * 40)
    print(f"{'np.float32':<20} {'torch.float32':<20}")
    print(f"{'np.float64':<20} {'torch.float64':<20}")
    print(f"{'np.int32':<20} {'torch.int32':<20}")
    print(f"{'np.int64':<20} {'torch.int64':<20}")
    print(f"{'np.uint8':<20} {'torch.uint8':<20}")

    numpy_array = np.array([[1.5, 2.5]], dtype=np.float32)
    tensor = torch.tensor(numpy_array)
    print(f"\n示例: np.float32 -> torch.float32")
    print(f"NumPy: {numpy_array.dtype} -> Tensor: {tensor.dtype}")


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch 与 NumPy 转换演示")
    print("=" * 60)

    demo_tensor_to_numpy()
    demo_from_numpy()
    demo_torch_tensor()
    demo_tensor_item()
    demo_shared_memory_warning()
    demo_numpy_to_tensor_dtype()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)

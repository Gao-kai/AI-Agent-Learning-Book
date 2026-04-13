"""
张量中元素类型转换

1. 直接在创建张量时指定数据类型 torch.tensor(dtype=...)
2. 对已创建张量tensor进行类型转换 tensor.type()
3. tensor.half() 将张量转换为float16类型
4. tensor.float() 将张量转换为float32类型
5. tensor.double() 将张量转换为float64类型
6. tensor.int() 将张量转换为int类型
7. tensor.short() 将张量转换为short类型
8. tensor.long() 将张量转换为long类型
9. tensor.bool() 将张量转换为bool类型
10. tensor.to() 通用类型转换方法 可以指定目标设备和数据类型

注意区分：
1. tensor.dtype 获取张量内部元素的类型 
2. type(tensor) 获取张量tensor数据类型
"""

import torch


def demo_dtype_at_creation():
    """
    方法1: 直接在创建张量时指定数据类型
    torch.tensor(data, dtype=torch.float32)
    - data: Python列表或数组
    - dtype: 目标数据类型
    返回: 指定类型的张量
    """
    print("\n========== 方法1: 创建时指定 dtype ==========")

    list_data = [[1, 2, 3], [4, 5, 6]]

    tensor_float = torch.tensor(list_data, dtype=torch.float32)
    print(f"torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)")
    print(f"参数: data=[[1,2,3],[4,5,6]], dtype=torch.float32")
    print(f"返回值:\n{tensor_float}")
    print(f"类型: {tensor_float.dtype}")

    tensor_int = torch.tensor(list_data, dtype=torch.int32)
    print(f"\ntorch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)")
    print(f"返回值:\n{tensor_int}")
    print(f"类型: {tensor_int.dtype}")

    tensor_bool = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    print(f"\ntorch.tensor([[1,0,1],[0,1,0]], dtype=torch.bool)")
    print(f"返回值:\n{tensor_bool}")
    print(f"类型: {tensor_bool.dtype}")


def demo_tensor_type_method():
    """
    方法2: 使用 tensor.type() 进行类型转换
    tensor.type(target_type)
    - target_type: 目标类型字符串，如 'torch.FloatTensor', 'torch.IntTensor'
    返回: 转换后的新张量
    """
    print("\n========== 方法2: tensor.type() ==========")
    tensor = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
    print(f"原始张量:\n{tensor}")
    print(f"原始类型: {tensor.dtype}")

    tensor_int = tensor.type(torch.IntTensor)
    print(f"\ntensor.type(torch.IntTensor)")
    print(f"返回值:\n{tensor_int}")
    print(f"类型: {tensor_int.dtype}")

    tensor_float = tensor_int.type(torch.FloatTensor)
    print(f"\ntensor_int.type(torch.FloatTensor)")
    print(f"返回值:\n{tensor_float}")
    print(f"类型: {tensor_float.dtype}")


def demo_tensor_half():
    """
    方法3: tensor.half() - 转换为 float16
    tensor.half()
    - 无参数
    返回: float16 类型的张量
    说明: float16 精度较低，但内存占用少，适合GPU加速
    """
    print("\n========== tensor.half() ==========")
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"原始张量:\n{tensor}")
    print(f"原始类型: {tensor.dtype}")

    tensor_half = tensor.half()
    print(f"\ntensor.half()")
    print(f"返回值:\n{tensor_half}")
    print(f"类型: {tensor_half.dtype}")
    print(f"内存占用: float32={tensor.element_size()}B, float16={tensor_half.element_size()}B")


def demo_tensor_float():
    """
    方法4: tensor.float() - 转换为 float32
    tensor.float()
    - 无参数
    返回: float32 类型的张量
    说明: float32 是 PyTorch 默认类型，适合深度学习训练
    """
    print("\n========== tensor.float() ==========")
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    print(f"原始张量:\n{tensor}")
    print(f"原始类型: {tensor.dtype}")

    tensor_float = tensor.float()
    print(f"\ntensor.float()")
    print(f"返回值:\n{tensor_float}")
    print(f"类型: {tensor_float.dtype}")


def demo_tensor_double():
    """
    方法5: tensor.double() - 转换为 float64
    tensor.double()
    - 无参数
    返回: float64 类型的张量
    说明: float64 精度最高，但内存占用大
    """
    print("\n========== tensor.double() ==========")
    tensor = torch.tensor([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=torch.float32)
    print(f"原始张量:\n{tensor}")
    print(f"原始类型: {tensor.dtype}")

    tensor_double = tensor.double()
    print(f"\ntensor.double()")
    print(f"返回值:\n{tensor_double}")
    print(f"类型: {tensor_double.dtype}")


def demo_tensor_int():
    """
    方法6: tensor.int() - 转换为 int32
    tensor.int()
    - 无参数
    返回: int32 类型的张量
    """
    print("\n========== tensor.int() ==========")
    tensor = torch.tensor([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=torch.float32)
    print(f"原始张量:\n{tensor}")
    print(f"原始类型: {tensor.dtype}")

    tensor_int = tensor.int()
    print(f"\ntensor.int()")
    print(f"返回值:\n{tensor_int}")
    print(f"类型: {tensor_int.dtype}")


def demo_tensor_short():
    """
    方法7: tensor.short() - 转换为 int16
    tensor.short()
    - 无参数
    返回: int16 类型的张量
    说明: short 是 int16 的别名，范围 -32768 ~ 32767
    """
    print("\n========== tensor.short() ==========")
    tensor = torch.tensor([[1000, 2000, 3000], [4000, 5000, 6000]], dtype=torch.int32)
    print(f"原始张量:\n{tensor}")
    print(f"原始类型: {tensor.dtype}")

    tensor_short = tensor.short()
    print(f"\ntensor.short()")
    print(f"返回值:\n{tensor_short}")
    print(f"类型: {tensor_short.dtype}")


def demo_tensor_long():
    """
    方法8: tensor.long() - 转换为 int64
    tensor.long()
    - 无参数
    返回: int64 类型的张量
    说明: long 是 int64 的别名，常用于索引和标签
    """
    print("\n========== tensor.long() ==========")
    tensor = torch.tensor([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=torch.float32)
    print(f"原始张量:\n{tensor}")
    print(f"原始类型: {tensor.dtype}")

    tensor_long = tensor.long()
    print(f"\ntensor.long()")
    print(f"返回值:\n{tensor_long}")
    print(f"类型: {tensor_long.dtype}")


def demo_tensor_bool():
    """
    方法9: tensor.bool() - 转换为 bool
    tensor.bool()
    - 无参数
    返回: bool 类型的张量
    说明: 非零值转为 True，零值转为 False
    """
    print("\n========== tensor.bool() ==========")
    tensor = torch.tensor([[1, 0, -1], [2, 0, -2]], dtype=torch.int32)
    print(f"原始张量:\n{tensor}")
    print(f"原始类型: {tensor.dtype}")

    tensor_bool = tensor.bool()
    print(f"\ntensor.bool()")
    print(f"返回值:\n{tensor_bool}")
    print(f"类型: {tensor_bool.dtype}")


def demo_tensor_to():
    """
    方法10: tensor.to() - 通用类型转换
    tensor.to(dtype=None, device=None, ...)
    - dtype: 目标数据类型
    - device: 目标设备
    - non_blocking: 是否异步传输
    - copy: 是否复制张量
    返回: 转换后的张量
    说明: to() 是最通用的转换方法，支持dtype和device同时转换
    """
    print("\n========== tensor.to() ==========")
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"原始张量:\n{tensor}")
    print(f"原始类型: {tensor.dtype}, 设备: {tensor.device}")

    tensor_int = tensor.to(dtype=torch.int32)
    print(f"\ntensor.to(dtype=torch.int32)")
    print(f"返回值:\n{tensor_int}")
    print(f"类型: {tensor_int.dtype}")

    tensor_half = tensor.to(dtype=torch.float16)
    print(f"\ntensor.to(dtype=torch.float16)")
    print(f"返回值:\n{tensor_half}")
    print(f"类型: {tensor_half.dtype}")


def demo_dtype_table():
    """
    PyTorch 数据类型对照表
    """
    print("\n========== PyTorch 数据类型对照表 ==========")
    print(f"{'方法':<20} {'dtype':<20} {'C类型':<15} {'说明'}")
    print("-" * 80)
    print(f"{'torch.float16 / half()':<20} {'torch.float16':<20} {'__half':<15} {'半精度浮点'}")
    print(f"{'torch.float32 / float()':<20} {'torch.float32':<20} {'float':<15} {'单精度浮点(默认)'}")  # noqa: E501
    print(f"{'torch.float64 / double()':<20} {'torch.float64':<20} {'double':<15} {'双精度浮点'}")
    print(f"{'torch.int8':<20} {'torch.int8':<20} {'int8_t':<15} {'8位整数'}")
    print(f"{'torch.int16 / short()':<20} {'torch.int16':<20} {'int16_t':<15} {'16位整数'}")
    print(f"{'torch.int32 / int()':<20} {'torch.int32':<20} {'int32_t':<15} {'32位整数'}")
    print(f"{'torch.int64 / long()':<20} {'torch.int64':<20} {'int64_t':<15} {'64位整数'}")
    print(f"{'torch.bool / bool()':<20} {'torch.bool':<20} {'bool':<15} {'布尔值'}")
    print(f"{'torch.uint8':<20} {'torch.uint8':<20} {'uint8_t':<15} {'无符号8位整数'}")

 
if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch 张量类型转换演示")
    print("=" * 60)

    demo_dtype_at_creation()
    demo_tensor_type_method()
    demo_tensor_half()
    demo_tensor_float()
    demo_tensor_double()
    demo_tensor_int()
    demo_tensor_short()
    demo_tensor_long()
    demo_tensor_bool()
    demo_tensor_to()
    demo_dtype_table()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)

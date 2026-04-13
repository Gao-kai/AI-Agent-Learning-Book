"""
创建线性张量
1. torch.linspace 创建等差数列线性张量
2. torch.arange 创建线性序列张量

设置随机种子
1. torch.random.initial_seed() 设置随机种子 默认 以时间戳为单位
2. torch.random.manual_seed()  手动设置随机种子 保证每次随机的结果相同

创建随机张量
3. torch.rand 创建随机浮点类型张量 [0,1) 区间内均匀分布
4. torch.randn 创建随机正态分布浮点类型张量 标准正态分布
5. torch.randint 创建随机整数类型张量

基于已有形状创建随机张量
6. torch.rand_like 创建与输入形状相同的随机浮点类型张量 [0,1) 区间内均匀分布
7. torch.randn_like 创建与输入形状相同的随机正态分布浮点类型张量 标准正态分布
"""

import torch


def demo_torch_linspace():
    """
    torch.linspace(start, end, steps)
    - start: 起始值
    - end: 结束值
    - steps: 元素数量
    返回: 等差数列张量
    """
    print("\n========== torch.linspace ==========")
    tensor = torch.linspace(start=0, end=10, steps=5)
    print(f"torch.linspace(start=0, end=10, steps=5)")
    print(f"参数: start=0, end=10, steps=5")
    print(f"返回值: {tensor}")
    print(f"类型: {tensor.dtype}")


def demo_torch_arange():
    """
    torch.arange(start, end, step)
    - start: 起始值（包含）
    - end: 结束值（不包含）
    - step: 步长
    返回: 线性序列张量
    """
    print("\n========== torch.arange ==========")
    tensor = torch.arange(start=0, end=10, step=2)
    print(f"torch.arange(start=0, end=10, step=2)")
    print(f"参数: start=0, end=10, step=2")
    print(f"返回值: {tensor}")
    print(f"类型: {tensor.dtype}")


def demo_random_initial_seed():
    """
    torch.random.initial_seed()
    - 无参数
    返回: 当前随机种子（默认基于时间戳）
    """
    print("\n========== torch.random.initial_seed() ==========")
    seed = torch.random.initial_seed()
    print(f"torch.random.initial_seed()")
    print(f"参数: 无")
    print(f"返回值: {seed} (当前随机种子)")


def demo_random_manual_seed():
    """
    torch.random.manual_seed(seed)
    - seed: 手动设置的随机种子（整数）
    返回: 无（设置全局随机种子）
    """
    print("\n========== torch.random.manual_seed() ==========")
    seed = 42
    torch.random.manual_seed(seed)
    print(f"torch.random.manual_seed(42)")
    print(f"参数: seed=42")
    tensor1 = torch.rand(3)
    
    torch.random.manual_seed(seed)
    tensor2 = torch.rand(3)
    print(f"第一次随机: {tensor1}")
    print(f"设置相同种子后再次随机: {tensor2}")
    print(f"两次结果相同: {torch.equal(tensor1, tensor2)}")


def demo_torch_rand():
    """
    torch.rand(*size)
    - size: 张量形状（如 2, 3 或 [2, 3]）
    返回: [0, 1) 均匀分布的随机浮点张量
    """
    print("\n========== torch.rand ==========")
    tensor = torch.rand(2, 3)
    print(f"torch.rand(2, 3)")
    print(f"参数: size=(2, 3)")
    print(f"返回值:\n{tensor}")
    print(f"类型: {tensor.dtype}")
    print(f"形状: {tensor.shape}")
    print(f"范围: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")


def demo_torch_randn():
    """
    torch.randn(*size)
    - size: 张量形状
    返回: 标准正态分布（均值=0, 标准差=1）的随机浮点张量
    """
    print("\n========== torch.randn ==========")
    tensor = torch.randn(2, 3)
    print(f"torch.randn(2, 3)")
    print(f"参数: size=(2, 3)")
    print(f"返回值:\n{tensor}")
    print(f"类型: {tensor.dtype}")
    print(f"形状: {tensor.shape}")
    print(f"均值: {tensor.mean().item():.4f}")
    print(f"标准差: {tensor.std().item():.4f}")


def demo_torch_randint():
    """
    torch.rand.randint(low, high, size)
    - low: 最小值（包含）
    - high: 最大值（不包含）
    - size: 张量形状
    返回: 随机整数张量
    """
    print("\n========== torch.randint ==========")
    tensor = torch.randint(low=0, high=10, size=(3, 3))
    print(f"torch.randint(low=0, high=10, size=(3, 3))")
    print(f"参数: low=0, high=10, size=(3, 3)")
    print(f"返回值:\n{tensor}")
    print(f"类型: {tensor.dtype}")
    print(f"形状: {tensor.shape}")
    print(f"范围: [{tensor.min().item()}, {tensor.max().item()}]")


def demo_torch_rand_like():
    """
    torch.rand_like(input)
    - input: 参考张量（浮点类型）
    返回: 与输入形状相同的 [0, 1) 均匀分布张量
    """
    print("\n========== torch.rand_like ==========")
    ref_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tensor = torch.rand_like(ref_tensor)
    print(f"参考张量:\n{ref_tensor}")
    print(f"参考张量类型: {ref_tensor.dtype}")
    print(f"torch.rand_like(参考张量)")
    print(f"返回值:\n{tensor}")
    print(f"形状: {tensor.shape}")


def demo_torch_randn_like():
    """
    torch.randn_like(input)
    - input: 参考张量（浮点类型）
    返回: 与输入形状相同的标准正态分布张量
    """
    print("\n========== torch.randn_like ==========")
    ref_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tensor = torch.randn_like(ref_tensor)
    print(f"参考张量:\n{ref_tensor}")
    print(f"参考张量类型: {ref_tensor.dtype}")
    print(f"torch.randn_like(参考张量)")
    print(f"返回值:\n{tensor}")
    print(f"形状: {tensor.shape}")


if __name__ == "__main__":
    print("=" * 50)
    print("PyTorch 张量创建方法演示")
    print("=" * 50)

    demo_torch_linspace()
    demo_torch_arange()
    demo_random_initial_seed()
    demo_random_manual_seed()
    demo_torch_rand()
    demo_torch_randn()
    demo_torch_randint()
    demo_torch_rand_like()
    demo_torch_randn_like()

    print("\n" + "=" * 50)
    print("演示完成")
    print("=" * 50)

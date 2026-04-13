"""
张量的基本运算

1. 加法 add
2. 减法 sub
3. 乘法 mul
4. 除法 div（可能返回小数）
5. 幂运算 pow
6. 指数运算 exp
7. 对数运算 log
8. 取整运算 floor
9. 取余运算 mod
10. 取反运算 neg

基本规则
- 所有张量和标量的基本运算，都会依次与张量中的每个元素进行运算
- 如果是标量，会自动广播到张量的形状

注意：
- 所有运算的方法加下划线比如add_ 表示该方法会修改原数据，类似于pandas的inplace=True
    - 例如：t1.add(10) 等价于 t2 = t1 + 10
    - 例如：t1.add_(10) 等价于 t1 += 10
- 所有张量的加减乘除运算，都可以直接通过符号+-*/ 来替代API调用
- 如果是张量和标量的运算，会自动广播标量到张量的形状（也就是标量会和张量中每个元素进行元素级运算）
"""

import torch

t = torch.tensor([1.0, 2.0, 3.0])

print("=" * 50)
print("1. 加法 (add)")
print("=" * 50)
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.tensor([4.0, 5.0, 6.0])
result = t1.add(t2)
print(f"t1 = {t1.tolist()}")
print(f"t2 = {t2.tolist()}")
print(f"t1.add(t2) = {result.tolist()}")
print(f"t1 + t2 = {(t1 + t2).tolist()}")
print(f"参数: other (Tensor) - 要相加的张量")
print(f"返回值: Tensor - 相加结果")

print("\n2. 减法 (sub)")
print(torch.tensor([4.0, 5.0, 6.0]) - torch.tensor([1.0, 2.0, 3.0]))

print("\n3. 乘法 (mul)")
print(torch.tensor([1.0, 2.0, 3.0]) * torch.tensor([4.0, 5.0, 6.0]))

print("\n4. 除法 (div)")
print(torch.tensor([4.0, 6.0, 8.0]) / torch.tensor([2.0, 3.0, 4.0]))

print("\n5. 幂运算 (pow)")
print(torch.tensor([2.0, 3.0, 4.0]) ** 2)

print("\n6. 指数运算 (exp)")
print(torch.exp(t))

print("\n7. 对数运算 (log)")
print(torch.log(t))

print("\n8. 取整运算 (floor)")
print(torch.floor(t))

print("\n9. 取余运算 (mod)")
print(torch.tensor([7.0, 8.0, 9.0]) % 3)

print("\n10. 取反运算 (neg)")
print(torch.neg(t))

print("\n" + "=" * 50)
print("in-place 操作示例 (带下划线)")
print("=" * 50)
t3 = torch.tensor([1.0, 2.0, 3.0])
t3.add_(10)
print(f"t3.add_(10) 会修改原张量: {t3.tolist()}")

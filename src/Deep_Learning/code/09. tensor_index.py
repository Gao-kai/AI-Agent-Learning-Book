"""
张量索引操作
操作张量的时候需要获取某些元素进行操作，这就需要基于索引进行操作。

1. 简单行列索引
2. 列表索引
3. 范围索引
4. 布尔索引
5. 多维索引
"""

# 导包
import torch

# 设置随机种子，确保结果可重复
torch.manual_seed(100);

# 创建一个3x5的随机整数张量
tensor = torch.randint(1,10,size=(4,6))
print(f"随机产生张量为：\n{tensor}")


"""
简单行列索引
1. 前行后列[行索引,列索引]
2. 冒号代表所有行/列
"""
def simple_index():
    # 获取第2行的元素
    print(tensor[1])
    print(tensor[1,:])

    # 获取第2列的元素
    print(tensor[:,1])

"""
列表索引
1. 列表索引中第一个列表中是行索引，第二个索引中是列索引
2. 当索引为一个列表的时候行列中元素数量必须一致
"""
def list_index():
    # 获取2行4列和3行5列的元素
    print(tensor[[1,2],[3,4]])

    # 获取第2行和第3行的1，2两列元素
    print(tensor[[[1],[2]],[1,2]])

"""
范围索引(类似Python中的切片操作)
1. 范围索引中可以使用冒号表示所有元素
2. 范围索引中可以使用切片操作
"""
def range_index():
    # 获取第2行的元素
    print(tensor[1])

    # 获取第2行的1-3列元素
    print(tensor[1,1:3])
    print(tensor[1:3,1:3])

    # 获取前3行的前2列数据
    # :3 表示获取索引为0，1，2行，不包含索引为3行
    # 3: 表示获取索引为3开始的所有行
    print(tensor[:3,:2])

    # 从第二行到最后一行的前2列数据
    print(tensor[1:,:2])

    # 获取所有索引奇数行 索引偶数列的元素
    # 1::2 表示开始索引为1，所有行/列，步长为2 ，获取索引为1，3，5行
    # 0::2 表示开始索引为0，所有行/列，步长为2 ，获取索引为0，2，4列
    print(tensor[1::2,0::2])



"""
布尔索引
1. 布尔索引中可以使用布尔值进行索引
"""

def bool_index():
    """
    获取第3列大于5的行数据(最终行可能被过滤 但是列一定不会少)
    1. 首先获取所有行的第3列元素 tensor[:,2]
    2. 然后基于张量运算获取第3列元素大于5布尔值张量 tensor[:,2] > 5 返回列元素组成布尔值张量[True,False,True,False,False]
    3. 最后做过滤求的是行数据 tensor[tensor[:,2] > 5] 那么将布尔张量放入行索引列表的位置即可逐行过滤

    """
    print(tensor[:,2] > 5)
    print(tensor[tensor[:,2] > 5])


    """
    获取第2行大于5的列数据（最终列可能被过滤 但是行一定不会少）
    1. 首先获取第2行的所有元素 tensor[1,:]
    2. 然后基于张量运算获取第2行元素大于5布尔值张量 tensor[1,:] > 5 返回行元素组成布尔值张量[True,False,True,False,False,True]
    3. 最后做过滤求的是列数据 tensor[:,tensor[1,:] > 5] 那么将布尔张量放入列索引列表的位置即可逐列过滤
    """
    print(tensor[1,:] > 5)
    print(tensor[1,:] > 5)
    print(tensor[:,tensor[1,:] > 5])

def multi_index():
    # 创建3维张量，Size为2*3*4
    tensor = torch.randint(1,10,size=(2,3,4))
    print(f"随机产生3维张量为：\n{tensor}")

    # 获取0轴上的第2个元素  0轴返回的是一个二维矩阵 0轴表示最高维度
    print(tensor[1,:,:])

    # 获取1轴上的第3个元素 1轴返回的是一个1维列表
    print(tensor[:,2,:])

    # 获取2轴上的第4个元素 2轴返回的是一个标量 也就是数组中元素
    print(tensor[:,:,3])

if __name__ == "__main__":
    # simple_index();
    # list_index()
    # range_index()
    # bool_index()
    multi_index()

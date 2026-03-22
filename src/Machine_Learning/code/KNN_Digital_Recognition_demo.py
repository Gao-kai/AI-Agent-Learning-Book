"""
KNN算法-手写数字识别案例

介绍：
    每张图片都由28*28像素组成,总计784个像素点，表示图片中每一个像素的颜色
    csv文件中第0列为label列 值为数字
    csv文件中第1-783列为特征数据 值为每一个像素点的像素值 范围为[0,255]


"""
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as pl # 导入Matplotlib库
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import warnings

# 消除来自sklearn.utils.deprecation警告
warnings.filterwarnings("ignore",module="sklearn");


# 获取当前脚本所在目录，构建数据文件的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "手写数字识别.csv")

# 获取index行数据的像素值并渲染出对应图像
# 参数：index 数字索引
# 返回值：读取索引为index行的像素 转化为28 * 28的像素后图像渲染
def show_digital_by_index(index):

    # 1. 加载数据集
    df = pd.read_csv(DATA_PATH)
    # print(f"数据集：{df}") # [42000 rows x 785 columns]

    # 2. 边界条件判断
    if index < 0 or index > len(df) - 1:
        print("索引超出范围")
        return

    # 3. 获取训练特征数据和训练标签数据
    # df.iloc[rowIndex,colIndex](integer location)方法是基于行索引和列索引来读取数据
    # rowIndex 行索引 从0开始 : 所有行 a:b 表示从第a行到第b行
    # colIndex 列索引 从0开始 
    # : 所有列 
    # a:b 表示从第a列到第b列
    x_train = df.iloc[:,1:] # 获取从第1列开始的所有数据
    y_train = df.iloc[:,0] # 获取第0列的数据

    # Counter方法统计每一个值出现多少次 返回一个类似JS中MAP
    print(f"训练标签数据分布情况：{Counter(y_train)}")

    # 4. 查看第index行的标签数据 -> 数字是多少？
    digital = y_train[index]
    print(f"第{index}行的像素渲染出的数字为：{digital}") 

    # 5. 查看第index行的特征数据 -> 数字对于的像素点的像素值是多少？
    # x_train.iloc[index]: 同时包含特征列名和对应数据
    # print(f"第{index}行的特征列名和数据为：{x_train.iloc[index]}")

    # x_train.iloc[index].values：只包含特征列数据
    # print(f"第{index}行的特征数据为：{x_train.iloc[index].values}")

    # 6. 将784个像素转化为28 * 28的像素后图像渲染
    # reshape方法：将一维数组数据转化为28行28列的矩阵
    pixel_data =  x_train.iloc[index].values.reshape(28,28);
    print(f"第{index}行的像素数据为：{pixel_data}")

    # 7. 绘制灰度图
    pl.imshow(pixel_data, cmap='gray')
    pl.axis('off')
    pl.show()


# 训练模型并保存至本地
def evaluate_knn_digital_recognition():
    # 1. 获取数据
    df = pd.read_csv(DATA_PATH)

    # 2. 数据预处理 拆分 + 归一化

    # 拆分出特征列
    x = df.iloc[:,1:]
    # 拆分出标签列
    y = df.iloc[:,0]
    print(f"数据集特征列：{x.shape}") # (42000, 784)
    print(f"数据集标签列：{y.shape}") # (42000, 1)
    print(f"数据集标签列的分布情况：{Counter(y)}")

    # 特征预处理的归一化
    x = x / 255

    # 拆分训练集和测试集
    # 参1 要拆分的特征列数据集x
    # 参2 要拆分的标签列数据集y
    # 参3 测试集所占比例
    # 参4 随机数种子
    # 参5 参考y数据集的分布进行抽取 保持随机抽取的数据中总是符合Counter(y)结果中0-9数字出现的比例恒定 防止出现抽取的数据中只包含0-8的数据 没有9的数据
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=18,stratify= y)


    # 3. 模型训练
    estimator = KNeighborsClassifier(n_neighbors=3);
    estimator.fit(x_train,y_train)

    # 4. 模型评估
    print(f"准确度为：{estimator.score(x_test,y_test)}")
    print(f"准确度为：{accuracy_score(y_test, estimator.predict(x_test))}")

    # 5. 模型保存
    joblib.dump(estimator, "../models/knn_digital_recognition_estimator.pkl")
    print(f"模型保存成功")


# 使用训练模型预测
def test_knn_digital_recognition_model():

    # 1. 本地加载训练的模型
    estimator = joblib.load("../models/knn_digital_recognition_estimator.pkl")

    # 2. 加载需要测试的图片 读取的结果为28* 28的二维数组
    x = pl.imread("../data/demo.png")
    print(f"图片像素数据为：{x.shape}") # (28, 28)

    # 转化二维数组为1*784的一维数组 
    # -1 表示自动计算列数 能转尽可能的转
    x = x.reshape(1, -1)

    # 训练时有归一化操作 所以这里也需要进行归一化 但是这里有个坑：
    # 如果pl.imread读取的图像返回值类型为uint8 才需要进行归一化操作
    # 其他类型比如float32等 已经自动归一化为0-1区间
    if x.dtype == np.uint8:
        x = x / 255
    print(f"图片像素数据为：{x.shape}-{x.dtype}") # (1, 784)-float32

    # 3. 调用模型预测
    y_pred= estimator.predict(x)
    print(f"预测结果为：{y_pred}")

    # 4. 绘制预测结果 查看是否正确
    pl.imshow(x.reshape(28,28), cmap='gray')
    pl.axis('off')
    pl.show()


test_knn_digital_recognition_model()
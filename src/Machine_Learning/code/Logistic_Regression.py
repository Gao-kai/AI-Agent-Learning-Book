"""
逻辑回归模型案例-疾病预测

逻辑回归模型：
1. 属于有监督学习算法，有特征有标签并且标签连续 适用于二分类问题
2. 逻辑回归原理：将线性回归得到的预测值通过Sigmod函数映射到0-1区间，然后预计设置的阈值结合概率进行分类
3. 逻辑回归损失函数：损失函数的作用就是衡量模型的预测能力，通常我们都希望损失函数的值最小，那么模型的误差越小，预测能力就越强。
                  因此逻辑回归的损失函数就是通过极大似然估计（求出一个模型参数让已观测到的样本出现的概率最高）的负数。
"""

import numpy as np
import pandas as pd
import os

from pydantic.experimental.pipeline import transform
from sklearn.linear_model import LogisticRegression # 逻辑回归模型
from sklearn.metrics import accuracy_score # 模型评估准确率
from sklearn.model_selection import train_test_split # 切分测试数据集
from sklearn.preprocessing import StandardScaler # 特征预处理-标准化

# 1. 加载数据
DIRNAME = os.path.dirname(__file__)
FILEPATH = os.path.join(DIRNAME,'..','data','breast-cancer-wisconsin.csv')
data = pd.read_csv(FILEPATH)
# data.info()

# 2. 数据预处理

# 将数据中异常值?转化为null值
data.replace('?', np.nan, inplace=True)

# 将包含null值的行进行删除
data.dropna(axis = 0,inplace=True, ignore_index = True)


# 3. 特征工程

# 3-1 特征提取
# 基于iloc函数来获取DataFrame中的特征列（获取所有行，以及从第2列到最后一列）
x = data.iloc[:,1:-1]
y = data.iloc[:,-1]

print(f'获取特征列数据为：{x.shape}')
print(f'获取标签列数据为：{y.shape}')
print(f'特征列数据前10行数据为：{x[:10]}')
print(f'标签列数据前10行数据为：{y[:10]}')

# 数据集切分为测试集和训练集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=22)

# 3-2 特征标准化
transformer = StandardScaler()
# 对训练集进行特征标准化
x_train = transformer.fit_transform(x_train)
# 对测试集进行特征标准化
x_test = transformer.transform(x_test)

# 4. 模型训练
estimator = LogisticRegression()
estimator.fit(x_train,y_train)

# 5. 模型预测
y_pred = estimator.predict(x_test)
print(f'预测结果为：{y_pred}')

# 6. 模型评估
# 准确率没有意义 因为逻辑回归的目的是二分类 不能说对于所有的测试集数据 90%的预测结果都是正确的 但是具体哪些正确哪些错误不能具体的话 就失去了模型评估的意义
print(f'模型对测试数据预测准确率为：{estimator.score(x_test,y_test)}')
print(f'模型对测试数据预测准确率为：{accuracy_score(y_test,y_pred)}')
"""
基于线性回归模型模拟正拟合
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
import matplotlib.pyplot as plt

# 1. 生产模拟数据

# 固定随机种子，确保虽然是随机数但是每次随机结果固定
np.random.seed(22)

# 生成100个-10到10之间的随机数 [1,2,3,...]
x = np.random.uniform(-3,3,100)

# 确定线性回归公式为 y = 2x^2 + 4x + 1 + ε（ε为随机噪声，服从标准正态分布，均值为0，标准差为1）
y = 2 * x ** 2 + 4 * x + 1 + np.random.normal(0,1,100) 


# 2. 数据预处理

# np的uniform方法生产出来的默认是一个一维数组，需要将其转化为n行1列的数据
# -1 表示自动计算行数 那么就是n行
# 1 表示列数为1列
X = x.reshape(-1,1)
X = np.hstack([X,X**2])
print(f"拼接后的特征数据前5行为：{X[:5]}")
print(f"当前的标签数据前5行为：{y[:5]}")

# 3. 特征工程（特征标准化）略

# 4. 模型训练
estimator = LinearRegression()
estimator.fit(X,y)

# 5. 模型预测
y_pred = estimator.predict(X)
print(f"模型预测值前5行：{y_pred[0:5]}")

# 6. 模型评估指标
print(f"均方误差：{mean_squared_error(y,y_pred)}")
print(f"均绝对误差：{mean_absolute_error(y,y_pred)}")
print(f"均方根误差：{root_mean_squared_error(y,y_pred)}")

# 7. 绘制真实散点图
plt.scatter(x,y)

# 8. 绘制预测值折线图
print(f"排序后的标签数据前5行为：{y_pred[:5]}")
print(f"排序后的原特征数据前5行为：{x[:5]}")

plt.plot(np.sort(x),y_pred[np.argsort(x)],color = 'red')
plt.show()



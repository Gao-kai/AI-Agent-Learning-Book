"""
线性回归模型预测加利福尼亚房价（随机梯度下降法）

1. 获取数据
2. 数据预处理
3. 特征工程（特征提取、特征归一化和标准化、特征降维）
4. 模型训练
5. 模型预测
6. 模型评估

线性回归算法模型：
- 属于监督学习
- 有特征有标签，并且标签是连续值

线性回归分类：
- 一元线性回归：1个特征列一个标签列
- 多元线性回归：多个特征列一个标签列

线性回归本质：线性回归模型其实就是通过线性公式来描述特征和标签之间的关系，并且通过损失函数求最小值来找到最佳的模型参数。
- 一元线性回归公式：y = w1 *x + b
- 多元线性回归公式：y = w1 *x1 + w2 *x2 + ... + wn *xn + b = w列向量的转置 * x + b

如何衡量线性回归模型的好坏？
- 思路：通过计算模型的损失函数值来衡量模型的好坏。
- 损失函数其实就是模型预测值与真实值之间的差异，也就是误差。

如何选择损失函数：

- 均方误差损失函数：MSE
- 均绝对误差损失函数：MAE
- 均方根误差损失函数：RMSE

如何让损失函数值最小？
- 正规方程法
- 梯度下降法

梯度下降有哪些方案？
- 全量梯度下降（FGD）
- 小批量梯度下降（Min-Batch 推荐）
- 随机梯度下降（SGD）
- 随机平均梯度下降（SAG）
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error

# 1. 获取数据集
data = fetch_california_housing(download_if_missing=True)
print(f"特征数据形状：{data.data.shape}")
print(f"标签数据形状：{data.target.shape}")
print(f"特征数据前5行：{data.data[:5]}")
print(f"标签数据前5行：{data.target[:5]}")
print(f"特征名称：{data.feature_names}")

# 2. 数据预处理
# 2.1 切分测试集和训练集

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=29
)

# 3. 特征工程（特征标准化）
# 3.1 创建标准化对象
transformer = StandardScaler()

# 3.2 对训练集和测试集的特征数据进行标准化处理
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)

# 4. 模型训练（SGDRegressor模型默认使用随机梯度下降法）
# fit_intercept=True 表示模型会自动计算截距项b
# loss="squared_error" 表示使用均方误差损失函数
# learning_rate="constant｜adaptive" 表示学习率固定或自适应调整
# eta0=0.01 表示学习率的初始值
# max_iter=1000 表示最大迭代次数
# tol=1e-3 表示停止条件，当模型参数变化小于1e-3时，认为模型收敛，停止迭代
# early_stopping=True 模型会自动从训练集中分出一部分作为验证集。当验证集上的性能不再提升时，训练就会停止
model = SGDRegressor(
    fit_intercept = True,
    loss="squared_error",
    learning_rate="constant", # 固定学习率
    eta0=0.0001,              # 固定模式下，0.0001 是更安全的起步值
    max_iter=2000,            # 增加迭代次数，因为步子小了，需要多走几步
    tol=1e-3,
    early_stopping=True,      # 依然开启早停，防止过拟合
    random_state=42
)
model.fit(X_train, y_train)
print(f"模型参数w：{model.coef_}")
print(f"模型截距项b：{model.intercept_}")


# 5. 模型预测
y_pred = model.predict(X_test)
print(f"测试集预测值前5行：{y_pred[:5]}")

# 6. 模型评估
print(f"测试集均方误差MSE：{mean_squared_error(y_test, y_pred)}")
print(f"测试集平均绝对误差MAE：{mean_absolute_error(y_test, y_pred)}")
print(f"测试集平均方根误差RMSE：{root_mean_squared_error(y_test, y_pred)}")

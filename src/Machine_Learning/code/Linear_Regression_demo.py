"""
线性回归模型API 示例
基于身高预测体重（一元线性回归）
"""

from sklearn.linear_model import LinearRegression
import numpy as np

# 1. 加载数据
x_train = np.array([[160], [166], [172], [174], [180]])     # 训练集特征
y_train = np.array([56.3, 60.6, 65.1, 68.5, 75])            # 训练集标签
x_test = np.array([[176]])                                  # 测试集特征


# 2. 数据预处理（跳过 不需要切分测试集和训练集 因为数据量少）
# 3. 特征工程（特征提取 & 特征预处理）
# 4. 模型训练
estimator = LinearRegression()
estimator.fit(x_train, y_train)
print(f"模型权重（斜率）: w: {estimator.coef_}")      #  w: [0.92942177]
print(f"模型偏置（截距）: b: {estimator.intercept_}") # b: -93.27346938775514

# 5. 模型评估
# 6. 模型预测
y_pred = estimator.predict(x_test)
print(f"模型预测值: y_pred: {y_pred}")  # y_pred: [70.3047619]



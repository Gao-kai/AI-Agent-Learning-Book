# 1. 导入必要的库
from sklearn.datasets import load_iris # 加载Iris数据集
from sklearn.model_selection import train_test_split, GridSearchCV # 导入训练-测试分割函数和网格搜索对象
from sklearn.neighbors import KNeighborsClassifier # 导入KNN分类器类
from sklearn.metrics import accuracy_score # 导入准确率评估指标
from sklearn.preprocessing import StandardScaler # 导入特征工程中特征标准化对象
import pandas as pd # 导入Pandas库
import seaborn as sns # 导入Seaborn库
import matplotlib.pyplot as pl # 导入Matplotlib库

# 2. 加载Iris数据集
def load_iris_dataset():
    iris_data = load_iris()
    # print(f"数据集信息：{iris_data}")
    print(f"数据集类型：{type(iris_data)}")
    print(f"数据集特征列：{iris_data.feature_names}") # ['sepal length', 'sepal width', 'petal length', 'petal width']
    print(f"数据集样本数据前10条:{iris_data.data[:10]}")
    print(f"数据集标签列前10条:{iris_data.target[:10]}")
    print(f"数据集标签列的枚举映射：{iris_data.target_names}") # ['setosa', 'versicolor', 'virginica']
    print(f"数据集基本描述：{iris_data.DESCR}")
    print(f"数据集的框架：{iris_data.frame}") # None
    print(f"数据集的文件名：{iris_data.filename}") # iris.csv
    print(f"数据集的模型包名：{iris_data.data_module}") # sklearn.datasets._base
    return iris_data
 

# 3. 绘制数据集的特征散点图
def plot_iris_dataset():
    # 加载Iris数据集
    iris_data = load_iris_dataset()

    # 封装数据集为 DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    
    # 添加标签label列
    iris_df['label'] = iris_data.target
    print(f"数据集特征列:\n{iris_df}")

    # 绘制特征散点图
    # data 参数：数据集
    # x 轴参数：特征列名称

    # hue 参数：按照label字段分组
    # fit_reg 参数：显示每一组的拟合回归线
    sns.lmplot(data = iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='label', fit_reg=True)
    
    # 设置标题 显示图像
    pl.title("Iris Dataset Feature Scatter Plot")
    pl.tight_layout ()
    pl.show()



# 4. 切分训练集和测试集的数据
def split_iris_dataset():
    # 加载Iris数据集
    iris_data = load_iris()

    # 数据预处理 - 切分测试集和训练集

    # 参数：test_size 测试集所占比例
    # 参数：random_state 随机数种子 随机数种子可以设置任意值 只要它不变 拆分结果就不会变 默认值为当前时间戳 每次运行都变 因此每次切分结果都不一致
    # 返回值：元组，分别是训练集特征和测试集的特征，训练集的标签和测试集的标签
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=15)

    print(f"训练集特征数据：{x_train},训练集特征个数：{len(x_train)}")
    print(f"训练集标签数据：{y_train},训练集标签个数：{len(y_train)}")
    print(f"测试集特征数据：{x_test},测试集特征个数：{len(x_test)}")
    print(f"测试集标签数据：{y_test},测试集标签个数：{len(y_test)}")



# 5. 利用KNN算法对鸢尾花进行分类 - 模型训练
def evaluate_knn_classifier():
    # 1. 加载Iris数据集
    iris_data = load_iris()

    # 2. 数据预处理 - 切分测试集和训练集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=23)
    print(f"测试集标签数据：{y_test},测试集标签个数：{len(y_test)}")
    # 3. 特征工程
    # 特征工程 - 提取（原数据集只有4个特征列且都为核心特征列 因此跳过特征提取）
    # 特征工程 - 预处理（特征标准化）

    # 3.1 创建特征标准化对象
    scaler = StandardScaler()

    # 3.2 对训练集特征数据进行标准化
    # fit_transform方法
    # 兼具fit训练和transform转换的功能 多用于第一次进行特征数据标准化的时候使用
    # 因为scaler会在第一次转化数据的同时按照给定的特征数据来调整自己的参数
    x_train_normalize = scaler.fit_transform(x_train)

    ## 3.3 对测试集特征数据进行标准化
    # transform方法：
    # 只有转化功能 因为首次进行数据标准化的时候已经训练过特征数据，因此只需要按照训练集的参数进行转换
    x_test_normalize = scaler.transform(x_test)


    # 4. 模型训练：基于KNN分类算法训练模型和超参数调优
    # 4.1 创建KNN分类器模型对象
    estimator = KNeighborsClassifier()

    # 4.2 定义超参数的取值范围
    param_grid = {'n_neighbors': [1,2,3,4,5,6,7]}

    # 4.3 创建网格搜索对象
    # 参数：estimator 模型对象
    # 参数：param_grid 超参数的取值范围为1-N
    # 参数：cv 交叉验证折数 假设为5折交叉验证 总计执行N*5次模型训练
    # 参数：scoring 评估指标
    estimator = GridSearchCV(estimator , param_grid, cv=5, scoring='accuracy')
    
    # 4.4 超参数调优后的模型训练过程 传入训练数据特征和训练数据标签
    estimator.fit(x_train_normalize,y_train)
    print(f"网格搜索最优超参数为：{estimator.best_params_}")
    print(f"网格搜索最优模型准确度为：{estimator.best_score_}")
    print(f"网格搜索最优模型评估器对象：{estimator.best_estimator_}")
    print(f"网格搜索具体交叉验证结果：{estimator.cv_results_}")

    # 4.5 选择网格搜索最优模型评估器对象为后续模型预测和评估的模型
    estimator = estimator.best_estimator_

    # 5. 模型预测
    # 5.1 场景1: 对已经切分且标准化后的测试集进行测试
    y_pred = estimator.predict(x_test_normalize)
    print(f"预测结果为：{y_pred}")

    # 5.2 场景2：对新数据进行预测
    new_data = [[7.8,2.1,3.9,1.6]]
    new_data_normalize = scaler.transform(new_data) # 新数据也需要进行特征数据标准化
    new_data_pred = estimator.predict(new_data_normalize) # 预测
    print(f"新数据预测结果为：{new_data_pred}")
    new_data_pred_proba = estimator.predict_proba(new_data_normalize) # 预测概率 其实就是查询新的数据最相似的前K个样本的分类概率 概率最大者胜出
    print(f"新数据预测概率为：{new_data_pred_proba}")


    # 6. 模型评估
    # 6.1 方法一：直接基于模型的score方法评估模型准确度
    score = estimator.score(x_train_normalize,y_train)
    print(f"基于训练集得出的模型准确度为：{score}")

    # 6.2 方法二：基于 accuracy_score方法评估模型准确度
    print(f"基于测试集的准确度为：{accuracy_score(y_test,y_pred)}")

evaluate_knn_classifier()










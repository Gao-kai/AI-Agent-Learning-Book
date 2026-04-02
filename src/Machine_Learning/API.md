# Python 常用 API 文档

## 1. pkl 文件详解

### 什么是 pkl 文件？

`.pkl` 是 Python **pickle** 模块生成的序列化文件格式。它用于将 Python 对象转换为字节流，以便存储到文件或通过网络传输。

### 为什么模型保存使用 pkl？

| 优势 | 说明 |
|------|------|
| **保留对象结构** | 序列化后的数据可以完整还原对象，包括类、方法、属性 |
| **跨平台兼容** | 字节流可在不同系统间传输 |
| **高效存储** | 二进制格式，比 JSON/XML 更紧凑 |
| **支持复杂对象** | 可序列化 NumPy 数组、scikit-learn 模型等复杂对象 |

### pickle vs joblib

```python
import pickle
import joblib

# pickle 方式（标准库）
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# joblib 方式（推荐用于机器学习模型）
joblib.dump(model, "model.pkl")
```

**joblib 优势**：
- 对大型 NumPy 数组优化更好
- 支持压缩（`compress=3`）
- 保存/加载速度更快

### 注意事项

```python
# 安全警告：不要加载不可信来源的 pkl 文件
# pickle 可以执行任意代码，存在安全风险
model = joblib.load("untrusted.pkl")  # 危险！
```

---

## 2. iloc API

### 简介

`iloc` 是 pandas 中基于**整数位置索引**的数据选择方法（integer location）。

### 基本语法

```python
df.iloc[行索引, 列索引]
```

### 常用示例

```python
import pandas as pd

df = pd.DataFrame({
    'A': [10, 20, 30, 40],
    'B': [100, 200, 300, 400],
    'C': [1000, 2000, 3000, 4000]
})
```

| 操作 | 代码 | 结果 |
|------|------|------|
| 选择第1行 | `df.iloc[0]` | 返回第0行的 Series |
| 选择最后1行 | `df.iloc[-1]` | 返回最后一行 |
| 选择第2行第1列 | `df.iloc[1, 0]` | 返回标量值 `20` |
| 选择前2行 | `df.iloc[0:2]` | 返回前两行 DataFrame |
| 选择前2行前2列 | `df.iloc[0:2, 0:2]` | 返回 2x2 DataFrame |
| 选择所有行的第1列 | `df.iloc[:, 0]` | 返回列A的所有值 |
| 选择第1、3行 | `df.iloc[[0, 2]]` | 返回指定行 |
| 选择第2、3列 | `df.iloc[:, 1:3]` | 返回列B和C |
| 选择特定行列 | `df.iloc[[0, 2], [0, 2]]` | 返回 (0,0), (2,2) 位置数据 |

### iloc vs loc

| 特性 | `iloc`          | `loc` |
|------|-----------------|-------|
| 索引方式 | 整数位置            | 标签名称 |
| 切片规则 | 左闭右开 `[0:2)`    | 左闭右闭 `[0:2]` |
| 示例 | `df.iloc[0, 1]` | `df.loc[0, 'B']` |

### 实际应用（手写数字识别）

```python
# 获取特征列（排除第0列的标签）
x_train = df.iloc[:, 1:]   # 所有行，第1列到最后一列

# 获取标签列（第0列）
y_train = df.iloc[:, 0]    # 所有行，第0列

# 获取特定行的特征数据
pixel_data = x_train.iloc[index].values
```

---

## 3. joblib.dump API

### 简介

`joblib` 是专门用于科学计算和机器学习的序列化工具，对 NumPy 数组有特殊优化。

### 安装

```bash
pip install joblib
```

### 核心方法

#### joblib.dump() - 保存模型

```python
import joblib
from sklearn.neighbors import KNeighborsClassifier

# 训练模型
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "model.pkl")

# 带压缩保存（压缩级别 0-9，默认 3）
joblib.dump(model, "model.pkl", compress=3)
```

#### joblib.load() - 加载模型

```python
# 加载模型
model = joblib.load("model.pkl")

# 使用模型预测
predictions = model.predict(X_test)
```

### 参数详解

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | object | 要序列化的 Python 对象 |
| `filename` | str | 保存路径 |
| `compress` | int/str | 压缩级别（0-9）或压缩算法 |
| `protocol` | int | pickle 协议版本 |

### 完整示例

```python
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 训练模型
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "../models/knn_model.pkl")
print("模型保存成功")

# 加载模型
loaded_model = joblib.load("../models/knn_model.pkl")

# 预测
result = loaded_model.predict(X_new)
```

### joblib vs pickle 性能对比

```python
import pickle
import joblib
import numpy as np
import time

# 创建大型数组
data = np.random.rand(10000, 1000)

# pickle 方式
start = time.time()
with open("data_pickle.pkl", "wb") as f:
    pickle.dump(data, f)
print(f"pickle 保存耗时: {time.time() - start:.3f}s")

# joblib 方式
start = time.time()
joblib.dump(data, "data_joblib.pkl", compress=3)
print(f"joblib 保存耗时: {time.time() - start:.3f}s")
```

---

## 4. matplotlib.pyplot API (imshow / imread)

### 简介

`matplotlib.pyplot` 是 Python 最常用的数据可视化库，`imshow` 和 `imread` 是图像处理的核心方法。

### 导入

```python
import matplotlib.pyplot as plt
```

### plt.imread() - 读取图像

```python
# 读取图像，返回 NumPy 数组
img = plt.imread("image.png")

print(img.shape)  # (height, width, channels)
print(img.dtype)  # uint8 或 float32
```

**返回值说明**：

| 图像类型 | shape | 值范围 |
|----------|-------|--------|
| 灰度图 | (H, W) | 0-255 或 0.0-1.0 |
| RGB 图 | (H, W, 3) | 0-255 或 0.0-1.0 |
| RGBA 图 | (H, W, 4) | 0-255 或 0.0-1.0 |

### plt.imshow() - 显示图像

```python
# 显示图像
plt.imshow(img)
plt.show()

# 显示灰度图
plt.imshow(img, cmap='gray')
plt.show()

# 显示热力图
plt.imshow(data, cmap='hot')
plt.colorbar()  # 添加颜色条
plt.show()
```

### imshow() 参数详解

| 参数 | 类型 | 说明 |
|------|------|------|
| `X` | array | 图像数据（MxN 或 MxNx3 或 MxNx4） |
| `cmap` | str | 颜色映射（'gray', 'hot', 'viridis' 等） |
| `vmin, vmax` | float | 颜色范围 |
| `aspect` | str | 宽高比（'auto', 'equal'） |
| `interpolation` | str | 插值方法（'nearest', 'bilinear' 等） |

### 常用 cmap 颜色映射

```python
# 灰度
plt.imshow(img, cmap='gray')

# 热力图
plt.imshow(data, cmap='hot')

# 冷暖色
plt.imshow(data, cmap='coolwarm')

# 彩虹色
plt.imshow(data, cmap='rainbow')
```

### 完整示例（手写数字识别）

```python
import matplotlib.pyplot as plt
import numpy as np

# 读取图像
img = plt.imread("../data/demo.png")
print(f"图像形状: {img.shape}")  # (28, 28)

# 转换为一维数组用于模型预测
img_flat = img.reshape(1, -1)
print(f"展平后形状: {img_flat.shape}")  # (1, 784)

# 显示图像
plt.imshow(img, cmap='gray')
plt.axis('off')  # 隐藏坐标轴
plt.title('Handwritten Digit')
plt.show()
```

### 其他常用方法

```python
# 隐藏坐标轴
plt.axis('off')

# 设置标题
plt.title('Image Title')

# 添加颜色条
plt.colorbar()

# 保存图像
plt.savefig('output.png', dpi=300, bbox_inches='tight')

# 创建子图
fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(img1, cmap='gray')
axes[0, 1].imshow(img2, cmap='gray')
plt.show()
```

---

## 5. reshape API

### 简介

`reshape` 是 NumPy 数组的方法，用于改变数组的形状（维度），**不改变数据本身**。

### 基本语法

```python
numpy.reshape(a, newshape, order='C')
# 或
array.reshape(newshape)
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `newshape` | 新形状，元组形式，如 `(28, 28)` 或 `(1, -1)` |
| `-1` | 自动计算该维度的大小 |
| `order` | 内存布局（'C' 行优先，'F' 列优先） |

### 常用示例

```python
import numpy as np

arr = np.arange(12)  # [0, 1, 2, ..., 11]
print(f"原始数组: {arr.shape}")  # (12,)

# 转换为 3x4 矩阵
matrix = arr.reshape(3, 4)
print(f"3x4 矩阵:\n{matrix}")

# 转换为 2x6 矩阵
matrix = arr.reshape(2, 6)

# 使用 -1 自动计算
matrix = arr.reshape(3, -1)  # 自动计算为 (3, 4)
matrix = arr.reshape(-1, 4)  # 自动计算为 (3, 4)

# 展平为一维数组
flat = matrix.reshape(-1)     # (12,)
flat = matrix.flatten()       # 返回副本
flat = matrix.ravel()         # 返回视图（可能）
```

### reshape vs flatten vs ravel

| 方法 | 返回值 | 内存 |
|------|--------|------|
| `reshape(-1)` | 一维数组 | 视图（可能） |
| `flatten()` | 一维数组 | 副本 |
| `ravel()` | 一维数组 | 视图（可能） |

### 实际应用（手写数字识别）

```python
# 784 个像素点 -> 28x28 图像
pixel_data = x_train.iloc[index].values  # (784,)
image = pixel_data.reshape(28, 28)        # (28, 28)

# 28x28 图像 -> 1x784 用于模型预测
img = plt.imread("../data/demo.png")      # (28, 28)
img_flat = img.reshape(1, -1)             # (1, 784)

# 批量处理
batch = images.reshape(100, -1)           # (100, 784)
```

### 注意事项

```python
# reshape 要求元素总数不变
arr = np.arange(12)
arr.reshape(3, 5)  # 错误！12 != 3*5

# 查看元素总数
print(arr.size)  # 12

# 正确做法
arr.reshape(3, 4)   # 12 == 3*4 ✓
arr.reshape(2, 6)   # 12 == 2*6 ✓
arr.reshape(4, 3)   # 12 == 4*3 ✓
```

---

## 快速参考表

| API | 用途 | 示例 |
|-----|------|------|
| `df.iloc[row, col]` | 按位置选择数据 | `df.iloc[:, 1:]` |
| `joblib.dump(obj, path)` | 保存模型 | `joblib.dump(model, "model.pkl")` |
| `joblib.load(path)` | 加载模型 | `model = joblib.load("model.pkl")` |
| `plt.imread(path)` | 读取图像 | `img = plt.imread("img.png")` |
| `plt.imshow(arr)` | 显示图像 | `plt.imshow(img, cmap='gray')` |
| `arr.reshape(shape)` | 改变数组形状 | `arr.reshape(28, 28)` |

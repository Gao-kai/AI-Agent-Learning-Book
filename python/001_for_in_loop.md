# Python vs TypeScript：for in 循环

## Python 解答

### 什么是 for in 循环？

`for in` 循环用于遍历可迭代对象（列表、字符串、字典等），依次取出每个元素进行处理。

### 基本用法

```python
# 遍历列表
fruits = ["苹果", "香蕉", "橙子"]
for fruit in fruits:
    print(fruit)

# 遍历字符串
for char in "Hello":
    print(char)

# 遍历字典
person = {"name": "张三", "age": 25}
for key in person:
    print(key, person[key])
```

### 带索引遍历

```python
fruits = ["苹果", "香蕉", "橙子"]
for index, fruit in enumerate(fruits):
    print(f"第{index+1}个水果是：{fruit}")
```

### 常用技巧

```python
# 循环指定次数
for i in range(5):
    print(i)  # 0 1 2 3 4

# 遍历多个列表（配对）
names = ["张三", "李四"]
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name}今年{age}岁")
```

## TypeScript 对比

### TypeScript 中的等价写法

```typescript
// 基本遍历
const fruits = ["苹果", "香蕉", "橙子"];
for (const fruit of fruits) {
    console.log(fruit);
}

// 带索引遍历
fruits.forEach((fruit, index) => {
    console.log(`${index + 1}: ${fruit}`);
});

// 循环指定次数
for (let i = 0; i < 5; i++) {
    console.log(i);
}
```

### 关键区别

| Python | TypeScript |
|--------|------------|
| `for item in iterable` | `for (const item of iterable)` |
| `enumerate()` | `forEach((item, index) => ...)` |
| `range(n)` | `for (let i = 0; i < n; i++)` |
| 遍历字典用 `in` | 遍历对象用 `for...in` |

## 最佳实践

1. **不要在循环中修改正在遍历的列表**

   ```python
   # ❌ 错误：会跳过元素
   numbers = [1, 2, 3]
   for num in numbers:
       if num == 2:
           numbers.remove(num)
   
   # ✅ 正确：创建副本
   for num in numbers.copy():
       if num == 2:
           numbers.remove(num)
   ```

2. **用 `enumerate()` 代替手动索引**

   ```python
   # ❌ 不推荐
   for i in range(len(fruits)):
       print(fruits[i])
   
   # ✅ 推荐
   for i, fruit in enumerate(fruits):
       print(fruit)
   ```

3. **用 `_` 表示不需要的变量**

   ```python
   for _ in range(3):
       print("重复3次")
   ```

4. **遍历字典时用 `items()` 获取键值对**

   ```python
   person = {"name": "张三", "age": 25}
   for key, value in person.items():
       print(f"{key}: {value}")
   ```

## 总结

Python 的 `for in` 循环语法简洁，支持多种可迭代对象。核心要点：用 `enumerate()` 带索引遍历，用 `zip()` 配对多个列表，用 `range()` 循环指定次数。

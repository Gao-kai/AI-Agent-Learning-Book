# `for await...of` 笔记

## 本质

`for await...of` 是 JavaScript 中用于遍历 **异步可迭代对象**（Async Iterable）的语法糖。

**核心本质**：它将异步数据流的逐个读取过程封装成了简洁的循环语法。

---

## 基本原理

### 1. 异步可迭代协议

一个对象要支持 `for await...of`，必须实现 **异步迭代协议**：

```typescript
interface AsyncIterable {
  [Symbol.asyncIterator](): AsyncIterator;
}

interface AsyncIterator {
  next(): Promise<{ value: any; done: boolean }>;
}
```

### 2. 工作流程

```
┌──────────────────────────────────────────────────────────┐
│                    执行流程                              │
├──────────────────────────────────────────────────────────┤
│  1. 调用 stream[Symbol.asyncIterator]() 获取迭代器        │
│                          ↓                              │
│  2. 调用 iterator.next() 返回 Promise                    │
│                          ↓                              │
│  3. await 这个 Promise 获取 { value, done }             │
│                          ↓                              │
│  4. 如果 done=false，将 value 赋值给 chunk 执行循环体     │
│                          ↓                              │
│  5. 重复步骤 2-4，直到 done=true                        │
└──────────────────────────────────────────────────────────┘
```

### 3. 手动实现等效代码

```typescript
// for await...of 的等效手动实现
const stream = await model.stream(prompt);
const iterator = stream[Symbol.asyncIterator]();

while (true) {
  const { value: chunk, done } = await iterator.next();
  if (done) break;
  // 循环体代码
  console.log(chunk);
}
```

---

## 为什么需要它？

| 同步迭代 `for...of` | 异步迭代 `for await...of` |
|---------------------|---------------------------|
| 遍历同步数据        | 遍历异步数据流            |
| 立即返回值          | 返回 Promise，需 await    |
| 阻塞式遍历          | 非阻塞式遍历              |

---

## 使用场景

| 场景 | 普通调用 | 流式调用 |
|------|---------|---------|
| **实时显示** | ❌ 等待全部返回 | ✅ 逐字显示，像打字机效果 |
| **长文本处理** | ❌ 需等待完整响应 | ✅ 边接收边处理，节省内存 |
| **进度反馈** | ❌ 无法显示进度 | ✅ 可显示加载进度 |

---

## 一个简单的自定义异步迭代器

```typescript
// 自定义异步可迭代对象
const asyncIterable = {
  async *[Symbol.asyncIterator]() {
    yield "Hello";
    await new Promise(r => setTimeout(r, 100));
    yield "World";
    await new Promise(r => setTimeout(r, 100));
    yield "!";
  }
};

// 使用 for await...of 遍历
for await (const value of asyncIterable) {
  console.log(value);
}
// 输出: Hello → World → !
```

---

## 总结

`for await...of` 的本质是 **异步迭代器的语法糖**，它自动处理 Promise 的等待和迭代状态的判断，让异步流式数据的处理变得简洁直观。

---

## 在 LangChain 中的应用

```typescript
// 流式调用大模型
const stream = await model.stream(prompt);
for await (const chunk of stream) {
  // 实时处理每个数据块
  process.stdout.write(chunk.content);
}
```

**注意**：并非所有模型都支持流式输出，需要根据实际情况选择调用方式。

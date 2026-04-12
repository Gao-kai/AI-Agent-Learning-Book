# LangChain Memory 管理策略学习笔记

## 目录

1. [为什么需要 Memory 管理](#1-为什么需要-memory-管理)
2. [三种 Memory 管理策略](#2-三种-memory-管理策略)
3. [存储层 API](#3-存储层-api)
4. [策略一：截断 (Truncation)](#4-策略一截断-truncation)
5. [策略二：总结 (Summarization)](#5-策略二总结-summarization)
6. [策略三：检索 (Retrieval)](#6-策略三检索-retrieval)
7. [核心 API 详解](#7-核心-api-详解)

---

## 1. 为什么需要 Memory 管理

大模型的上下文大小总是有限的，当对话历史无限增长时，最终会超出模型上限。

```
问题：无限增长的对话历史 → 超出上下文限制 → 无法继续对话
```

### 手动管理 Message 数组的缺点

- 需要手动管理 Message 数组
- Message 数组可能超出模型上限
- 无法自动化处理

---

## 2. 三种 Memory 管理策略

| 策略 | 核心思想 | 实现方式 |
|------|----------|----------|
| **截断** | 保留最近的消息，舍弃旧消息 | 根据 Token 数量截断 |
| **总结** | 对旧消息生成摘要，只保留摘要和最近消息 | 调用大模型生成摘要 |
| **检索** | 将历史存入向量数据库，按需检索 | RAG 语义检索 |

```
┌─────────────────────────────────────────────────────────┐
│                     Memory 管理架构                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌───────────────┐                                     │
│   │  存储层       │  → InMemoryChatMessageHistory        │
│   │  (数据存储)   │  → FileSystemChatMessageHistory      │
│   │               │  → Milvus (向量数据库)               │
│   └───────┬───────┘                                     │
│           │                                             │
│   ┌───────┴───────┐                                     │
│   │  逻辑层       │                                     │
│   │  (管理策略)   │                                     │
│   └───────────────┘                                     │
│                                                         │
│   - 截断 (Truncation)  - trimMessages                   │
│   - 总结 (Summarization) - getBufferString + LLM        │
│   - 检索 (Retrieval)   - Milvus RAG                      │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 存储层 API

### 3.1 InMemoryChatMessageHistory（内存存储）

适用于**短期记忆**，数据存储在内存中，程序结束后消失。

```typescript
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";

const chatHistory = new InMemoryChatMessageHistory();

// 添加消息
await chatHistory.addMessage(new HumanMessage("你好"));
await chatHistory.addMessage(new AIMessage("你好，有什么可以帮助你？"));

// 获取所有消息
const messages = await chatHistory.getMessages();

// 清空
await chatHistory.clear();
```

### 3.2 FileSystemChatMessageHistory（文件系统存储）

适用于**长期记忆**，数据持久化到 JSON 文件中。

```typescript
import { FileSystemChatMessageHistory } from "@langchain/community/stores/message/file_system";
import { fileURLToPath } from "url";
import path from "path";

// 获取文件路径（ES Module 写法）
const currFilePath = fileURLToPath(import.meta.url);
const __dirname = path.dirname(currFilePath);
const chatMessagesFilePath = path.join(__dirname, "./chat_history.json");

const chatHistory = new FileSystemChatMessageHistory({
  sessionId: "user_123",
  userId: "user_123",
  filePath: chatMessagesFilePath,
});

// 使用同上，API 与 InMemoryChatMessageHistory 一致
await chatHistory.addMessage(new HumanMessage("你好"));
const messages = await chatHistory.getMessages();
```

> ⚠️ **注意**：`import.meta.url` 是 `file://` URL 格式，需要用 `fileURLToPath` 转换。

### 3.3 短期记忆 vs 长期记忆

| 类型 | 存储位置 | 生命周期 | 示例 |
|------|----------|----------|------|
| **短期记忆 (STM)** | 内存 | 程序运行期间 | InMemoryChatMessageHistory |
| **长期记忆 (LTM)** | 磁盘/数据库 | 持久化 | FileSystemChatMessageHistory、Milvus |

---

## 4. 策略一：截断 (Truncation)

### 4.1 原理

当消息超过指定阈值时，**保留最近的消息，舍弃旧消息**。

```
原始消息：[M1, M2, M3, M4, M5, M6, M7, M8]
阈值：保留最近 6 条

截断后：[M3, M4, M5, M6, M7, M8]  ← 舍弃 M1, M2
```

### 4.2 按消息数量截断

```typescript
const maxMessageCount = 6;
const trimmedMessage = totalMessages.slice(-maxMessageCount);
```

### 4.3 按 Token 数量截断（推荐）

使用 `trimMessages` API，可以更精确地控制上下文长度。

```typescript
import { trimMessages } from "@langchain/core/messages";
import { getEncoding, getEncodingNameForModel } from "js-tiktoken";

const trimmedMessage = await trimMessages(totalMessages, {
  maxTokens: 120,                            // 最大 token 数
  strategy: "last",                          // 保留最新的
  tokenCounter: (msgs) => countTokens(msgs), // 自定义 token 计算器
  includeSystem: true,                       // 保留系统消息
});
```

#### 自定义 Token 计算器

```typescript
function countTokens(messages: BaseMessage[]) {
  const encodingName = getEncodingNameForModel("gpt-4");
  const encodingInstance = getEncoding(encodingName);
  let totalTokens = 0;

  for (const message of messages) {
    const content = typeof message.content === "string"
      ? message.content
      : JSON.stringify(message.content);
    totalTokens += encodingInstance.encode(content).length;
  }
  return totalTokens;
}
```

---

## 5. 策略二：总结 (Summarization)

### 5.1 原理

当消息超过阈值时：

1. 保留最近 N 条消息
2. 对旧消息调用大模型生成摘要
3. 用摘要替代旧消息

```
原始消息：[M1, M2, M3, M4, M5, M6]
Token: 500（超过阈值 200）

保留最近：[M5, M6]  Token: 80（不超过阈值 80）
需要总结：[M1, M2, M3, M4]  Token: 420

总结后：[SystemMessage(摘要), M5, M6]  Token: 150（符合要求）
```

### 5.2 实现代码

```typescript
import { getBufferString } from "@langchain/core/messages";

async function summarizeHistoryMessage(messages) {
  // 将消息数组转为字符串
  const historyMessagesToText = getBufferString(messages, "用户", "AI助手");

  const summaryPrompt = `
    请你总结以下对话的核心内容，保留重要信息:
    ${historyMessagesToText}
  `;

  const response = await model.invoke([new SystemMessage(summaryPrompt)]);
  return response.content;
}

// 判断是否需要总结
if (totalTokens > MAX_TOKEN) {
  // 1. 分离需要保留和需要总结的消息
  let toStoredMessages = [];
  let storedTokens = 0;

  for (let i = totalMessages.length - 1; i >= 0; i--) {
    const message = totalMessages[i];
    const messageToken = countToken(message);
    if (storedTokens + messageToken <= MAX_STORED_TOKEN) {
      toStoredMessages.unshift(message);  // 头部入栈，保持顺序
      storedTokens += messageToken;
    } else {
      break;
    }
  }

  // 2. 总结旧消息
  const toSummarizeMessages = totalMessages.slice(
    0,
    totalMessages.length - toStoredMessages.length
  );
  const summaryMessage = await summaryHistoryMessage(toSummarizeMessages);

  // 3. 清空并重新添加
  await history.clear();
  await history.addMessage(new SystemMessage(summaryMessage));
  for (const message of toStoredMessages) {
    await history.addMessage(message);
  }
}
```

### 5.3 实际应用场景

类似于 Cursor 等 AI 编程工具，当上下文快达到限制时，**自动触发总结**。

---

## 6. 策略三：检索 (Retrieval)

### 6.1 原理

将历史消息存入**向量数据库**，当需要时通过**语义检索**找到相关历史。

```
用户: "我之前提到的机器学习进度怎么样了？"
         ↓
  语义检索（向量化）
         ↓
  从 Milvus 找到最相关的历史对话
         ↓
  将相关历史 + 当前问题 → 发送给 LLM
```

### 6.2 完整流程

```typescript
async function retrievalMemory(client: MilvusClient) {
  const userInput = "我之前提到的机器学习的进度怎么样了?";

  // 1. 从向量数据库检索相关历史
  const searchResult = await retrievalHistoryMessage(client, userInput, 2);

  // 2. 格式化检索结果
  let relevantHistory = searchResult
    .map((res) => `历史会话: ${res.content}`)
    .join("\n\n");

  // 3. 构建上下文
  const contextMessages = relevantHistory
    ? [new HumanMessage(relevantHistory), new HumanMessage(userInput)]
    : [new HumanMessage(userInput)];

  // 4. 调用 LLM
  const response = await llm.invoke(contextMessages);

  // 5. 保存新对话到向量数据库
  await client.insert({
    collection_name: COLLECTION_NAME,
    data: [{
      id: `conv_${Date.now()}`,
      content: `用户: ${userInput}\n助手: ${response.content}`,
      vector: await getEmbedding(`用户: ${userInput}\n助手: ${response.content}`),
      round: queryAllData.length + 1,
      timestamp: new Date().toISOString(),
    }],
  });
}
```

### 6.3 Milvus CRUD 操作

```typescript
// 1. 连接
const client = new MilvusClient({ address: "127.0.0.1:19530" });
await client.connectPromise;

// 2. 创建集合
await client.createCollection({
  collection_name: "chat_history",
  fields: [
    { name: "id", data_type: DataType.VarChar, max_length: 50, is_primary_key: true },
    { name: "vector", data_type: DataType.FloatVector, dim: 1024 },
    { name: "content", data_type: DataType.VarChar, max_length: 5000 },
  ],
});

// 3. 创建索引
await client.createIndex({
  collection_name: "chat_history",
  field_name: "vector",
  index_type: IndexType.IVF_FLAT,
  metric_type: MetricType.COSINE,
});

// 4. 加载集合
await client.loadCollection({ collection_name: "chat_history" });

// 5. 插入数据
await client.insert({
  collection_name: "chat_history",
  data: [{ id: "001", vector: [...], content: "..." }],
});

// 6. 搜索
const results = await client.search({
  collection_name: "chat_history",
  vector: queryVector,
  limit: 2,
  metric_type: MetricType.COSINE,
});

// 7. 查询所有
await client.query({ collection_name: "chat_history", filter: "" });

// 8. 删除
await client.delete({ collection_name: "chat_history", filter: 'id == "001"' });
```

---

## 7. 核心 API 详解

### 7.1 js-tiktoken（Token 计算）

| API | 说明 |
|-----|------|
| `getEncodingNameForModel("gpt-4")` | 获取模型对应的编码名称 |
| `getEncoding("cl100k_base")` | 获取编码对象 |
| `encoding.encode(text)` | 计算 token 数量 |

```typescript
const encodingName = getEncodingNameForModel("gpt-4");
const encoding = getEncoding(encodingName);
const tokens = encoding.encode("你好，世界");
console.log(tokens.length); // token 数量
```

### 7.2 trimMessages（消息截断）

```typescript
await trimMessages(messages, {
  maxMessages: 6,           // 最大消息数量
  maxTokens: 500,          // 或最大 token 数
  strategy: "last",        // "first" 保留最早，"last" 保留最新
  tokenCounter: fn,        // 自定义 token 计算
  includeSystem: true,     // 是否保留系统消息
  startOn: "human",        // 从什么类型开始
  endOn: "ai",             // 以什么类型结束
});
```

### 7.3 getBufferString（消息转字符串）

```typescript
const str = getBufferString(messages, "用户", "AI助手", "系统");
// 输出：
// 系统: 你是一个助手
// 用户: 你好
// AI助手: 你好，有什么可以帮助你？
```

### 7.4 Milvus 索引参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `index_type` | 索引类型 | `IVF_FLAT`（中等数据）、`HNSW`（大数据） |
| `metric_type` | 相似度计算 | `COSINE`（余弦）、`L2`（欧氏距离）、`IP`（内积） |
| `nlist` (IVF) | 聚类中心数量 | 1024~16384 |
| `nprobe` (搜索) | 搜索范围 | 16~64，越大越精准越慢 |

---

## 总结对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **截断** | 简单、实现成本低 | 丢失历史信息 | 短期对话、快速迭代 |
| **总结** | 保留核心信息 | 消耗额外 token、可能有信息损失 | 中长期项目、复杂上下文 |
| **检索** | 完整保留、按需获取 | 需要额外存储、检索有延迟 | 长期知识库、复杂多轮对话 |

"""
Langchain中的上下文管理与记忆系统

# 基础概念
由于大模型默认是不具有历史消息的记忆能力的，因此要让Agent享有记忆，就必须基于Langchain的上下文工程来进行管理

# 上下文工程
上下文工程就是负责合理组织记忆的工程，从而让LLM的响应更加连贯，也是Agent实现多轮交互的核心基础。
1. 动态运行时上下文 对应的是短期记忆 也就是提供单次会话的历史记录 通过LangGraph中的state对象实现
2. 动态跨会话上下文 对应的是长期记忆 也就是提供跨对话级别的持久记录 通过LangGraph中的store对象实现
3. 静态运行时上下文 对应的是Agent启动时传入的context对象 只在单次运行生效 LangGraph中的context对象实现

# 记忆分类
1. 短期记忆 Short-term Memory
会话级别记忆，只在单个会话线程中有效，一旦开启新的会话thread id改变 记忆消失

2. 长期记忆 Long-term Memory
跨会话级别记忆 在会话期间存储历史数据，在线程中共享，可以被任意线程调用


# InMemorySaver 与 PostgresSaver 的区别

## 核心区别对比
| 维度 | InMemorySaver | PostgresSaver |
|------|---------------|---------------|
| 存储位置 | 进程内存（RAM） | PostgreSQL 数据库 |
| 数据持久化 | ❌ 进程退出即丢失 | ✅ 永久保存 |
| 跨进程共享 | ❌ 仅当前进程 | ✅ 多进程/多实例共享 |
| 重启恢复 | ❌ 重启后清空 | ✅ 重启后恢复 |
| 内存占用 | ⭐⭐⭐ 高（随消息增长） | ⭐⭐ 低（数据在数据库） |
| 性能 | ⭐⭐⭐⭐⭐ 最快 | ⭐⭐⭐⭐ 较快（网络开销） |
| 使用复杂度 | ⭐ 简单 | ⭐⭐⭐ 需要配置数据库 |
| 生产环境适用 | ❌ 仅开发/测试 | ✅ 生产环境 |

## 存储原理

### InMemorySaver：内存字典
内部就是一个 Python dict，数据存储在进程内存中

### PostgresSaver：数据库表
将数据写入 PostgreSQL 的 checkpoints 表，通过 SQL 查询读取

## 生命周期对比

### InMemorySaver 的生命周期
进程启动 → 数据写入内存 → 进程退出 → 数据全部丢失

### PostgresSaver 的生命周期
进程启动 → 连接数据库 → 数据写入表 → 进程退出 → 数据仍然存在

## 使用场景

### 选择 InMemorySaver 当：
- 快速原型开发
- 本地测试
- 不需要持久化的场景
- 对性能要求极高的短期任务

### 选择 PostgresSaver 当：
- 生产环境部署
- 需要跨服务共享对话状态
- 需要重启后恢复对话
- 需要数据持久化和备份
- 需要多实例水平扩展

## 代码层面的区别

### InMemorySaver：直接创建实例
```python
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()  # 直接创建，不需要 with 块

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=saver,  # 直接传入
)
```

### PostgresSaver：需要 with 块管理
```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(conn_string) as saver:
    saver.setup()  # 需要初始化表
    
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=saver,  # 在 with 块内传入
    )
    # 所有操作都必须在 with 块内
```

## 其他持久化 Saver
| Saver | 存储方式 | 特点 |
|-------|----------|------|
| RedisSaver | Redis | 高性能缓存，适合会话存储 |
| SqliteSaver | SQLite | 轻量级文件数据库 |
| PostgresSaver | PostgreSQL | 企业级生产环境 |
| InMemorySaver | 内存 | 开发测试专用 |

## 总结
开发阶段: InMemorySaver → 快速迭代，无需配置
生产阶段: PostgresSaver → 持久化，高可用，可扩展

一句话概括：InMemorySaver 是临时草稿纸，PostgresSaver 是正式文件柜。
草稿纸方便但丢了就没了，文件柜安全但需要管理。
"""

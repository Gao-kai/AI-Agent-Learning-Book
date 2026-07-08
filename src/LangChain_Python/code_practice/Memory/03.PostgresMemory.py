"""
Postgres 长期记忆

# 基础
使用 langgraph-checkpoint-postgres 替代 InMemorySaver，实现对话历史的持久化存储
适用于生产环境，支持跨进程共享，重启不丢失

# 安装依赖
pip install langgraph-checkpoint-postgres

# 实现步骤
1. 安装 langgraph-checkpoint-postgres 包
2. 配置 PostgreSQL 数据库连接字符串
3. 使用 PostgresSaver.from_conn_string() 创建检查点
4. 在 with 块内完成所有操作（setup/compile/invoke）
5. 使用 thread_id 标识唯一会话

# 注意事项
1. PostgresSaver.from_conn_string() 返回的是上下文管理器，必须在 with 块内使用
2. 所有依赖 saver 的逻辑必须放进同一个 with 块内
3. 首次使用需要调用 saver.setup() 创建数据库表

# PostgresSaver 的执行流程
agent.invoke() 时的完整工作流程：

1. 根据 thread_id 从 PostgreSQL 查询历史消息
   SELECT checkpoint FROM checkpoints WHERE thread_id = '001'

2. 组装完整消息列表
   [历史消息1, 历史消息2, ..., 新用户消息]

3. 发送给大模型
   model.generate(messages)

4. 获取模型响应
   AI: 大熊猫基地。

5. 更新数据库
   INSERT INTO checkpoints (thread_id, checkpoint)
   VALUES ('001', {'messages': [..., AI回复]})

# 核心原理
PostgresSaver = 数据库版的 InMemorySaver
工作流程完全一样，只是把内存换成了数据库：
- 代价：每次 invoke 需要数据库读写（毫秒级）
- 换来：持久化、跨进程共享、重启恢复

# 关键区别对比
| 特性 | InMemorySaver | PostgresSaver |
|------|---------------|---------------|
| 读取位置 | 本地内存 | 远程数据库 |
| 写入位置 | 本地内存 | 远程数据库 |
| 读取速度 | 极快（纳秒级） | 较快（毫秒级） |
| 写入速度 | 极快（纳秒级） | 较快（毫秒级） |
| 持久化 | ❌ | ✅ |
| 跨进程 | ❌ | ✅ |

# 设计思想
状态持久化 + 按需加载：
- 每次对话结束后，状态被保存到数据库，不会丢失
- 每次对话开始时，只加载当前 thread_id 的状态，不是全量加载

# 缺点与解决方案
默认情况下，历史消息会无限累积，导致 Token 爆炸：
- 解决方案1：消息截断（trim_messages）
- 解决方案2：消息摘要（create_summarization_tool_middleware）
- 解决方案3：混合策略（推荐，见 04.Memory_Manage.py）
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages.human import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from src.LangChain_Python.models.chat_model import model
from rich import print as rprint


def main():
    load_dotenv(override=True)
    conn_string = os.getenv("POSTGRES_DB_URL")
    
    rprint("连接字符串:", conn_string)
    
    if not conn_string:
        rprint("❌ 环境变量 POSTGRES_DB_URL 未设置！")
        return

    with PostgresSaver.from_conn_string(conn_string) as saver:
        rprint("✅ 连接成功，开始创建表...")
        saver.setup()
        rprint("✅ 表创建成功！")

        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="假设你是一名智能的AI旅游助手",
            checkpointer=saver,
        )

        config = {"configurable": {"thread_id": "001"}}

        response = agent.invoke(
            {
                "messages": [HumanMessage("成都最著名的一个景点是什么？只说一个就好")],
            },
            config=config,
        )

        rprint("第一次对话响应:", response["messages"][-1].content)

        response = agent.invoke(
            {
                "messages": [HumanMessage("我刚才问了什么问题？")],
            },
            config=config,
        )

        rprint("第二次对话响应:", response["messages"][-1].content)

        state = agent.get_state(config=config)
        rprint("\n历史消息:")
        for msg in state.values["messages"]:
            rprint(msg)


if __name__ == "__main__":
    main()

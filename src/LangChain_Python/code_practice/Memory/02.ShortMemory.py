"""
短期记忆

# 基础
Langchain中短期记忆是三者的结合：
1. State 会话内部状态 保存着messages历史消息对象
2. CheckPointer 负责将state对象作为检查点持久化保存，一个检查点是一个时刻的state快照
3. Thread ID 用于标识State的唯一ID

# 实现
1. 初始化检查点 checkpointer 创建唯一的内存级别的记忆存储
2. 绑定Agent 在创建Agent的时候传入checkpointer = InMemorySaver
3. 设定唯一会话ID
"""

import os
from typing import Literal

from langchain.agents import create_agent
from IPython.display import Image, display
from langchain_core.messages.human import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from src.LangChain_Python.models.chat_model import model
from rich import print as rprint

# 创建Agent
"""
checkpointer做了什么？
1. 取出state对象的历史消息
2. 追加新的消息
3. 调用模型传入历史消息
4. 保存模型返回的消息，更新历史列表

注意：
1. InMemorySaver只在同一进程有效，进程重启丢失，也不支持跨进程共享
2. InMemorySaver中保存的消息会一直增长，因此需要管理上下文，否则token会快速增长
"""
agent = create_agent(
    model=model,
    tools=[],
    system_prompt="假设你是一名智能的AI旅游助手",
    checkpointer=InMemorySaver(),
)

# 创建会话ID
config = {"configurable": {"thread_id": "001"}}


# 调用Agent
response = agent.invoke(
    {
        "messages": [HumanMessage("成都最著名的一个景点是什么？只说一个就好")],
    },
    config=config,
)


# 再次调用Agent
response = agent.invoke(
    {
        "messages": [HumanMessage("我刚才问了什么问题？")],
    },
    config=config,
)

# 查看内存中保存的历史消息
state = agent.get_state(config=config)
for msg in state.values["messages"]:
    rprint(msg)

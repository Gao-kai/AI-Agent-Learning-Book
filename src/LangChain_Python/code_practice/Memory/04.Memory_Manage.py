"""
记忆管理：消息摘要与截断

# 问题
默认情况下，每次 invoke 都会把所有历史消息发送给大模型，
导致消息无限累积，Token 消耗越来越大，最终超出上下文窗口。

# 解决方案
- 解决方案1：消息截断（trim_messages）在模型调用之前截断消息，控制大模型看到的上下文的宽度，基于@before_model实现
- 解决方案2：消息删除（remove_messages）在模型调用之后删除多余消息，基于@after_model实现
            - RemoveMessage执行的时候并不会真的删除数据，而是做一个标记，告诉框架某个信息是删除的。
              再次再调用的时候 就会把原始消息和标记的消息放在一起计算，如果id一样，说明标记数据需要删除
- 解决方案3：消息摘要（Summarize 中间件实现）对历史消息进行摘要，用摘要代替详细历史

# 适用场景
- 长时间对话的 Agent
- 需要控制 Token 消耗的场景
- 生产环境部署
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from src.LangChain_Python.models.chat_model import model
from rich import print as rprint


def truncation_example():
    """消息截断示例：只保留最近5条消息"""
    load_dotenv(override=True)
    conn_string = os.getenv("POSTGRES_DB_URL")

    with PostgresSaver.from_conn_string(conn_string) as saver:
        saver.setup()

        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="假设你是一名智能的AI旅游助手",
            checkpointer=saver,
        )

        config = {"configurable": {"thread_id": "truncation_test"}}

        messages_to_send = [HumanMessage("成都最著名的一个景点是什么？")]
        response = agent.invoke({"messages": messages_to_send}, config=config)
        rprint("第1次对话:", response["messages"][-1].content)

        messages_to_send = [HumanMessage("门票多少钱？")]
        response = agent.invoke({"messages": messages_to_send}, config=config)
        rprint("第2次对话:", response["messages"][-1].content)

        messages_to_send = [HumanMessage("在哪里？")]
        response = agent.invoke({"messages": messages_to_send}, config=config)
        rprint("第3次对话:", response["messages"][-1].content)

        state = agent.get_state(config=config)
        all_messages = state.values["messages"]
        rprint(f"\n原始消息数量: {len(all_messages)}")

        truncated_messages = trim_messages(all_messages, max_messages=2)
        rprint(f"截断后消息数量: {len(truncated_messages)}")


def summarization_middleware_example():
    """消息摘要示例：使用中间件自动摘要"""

    load_dotenv(override=True)
    conn_string = os.getenv("POSTGRES_DB_URL")

    summarization_middleware = create_summarization_tool_middleware(
        model=model,
        max_tokens=500,
    )

    with PostgresSaver.from_conn_string(conn_string) as saver:
        saver.setup()

        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="假设你是一名智能的AI旅游助手",
            checkpointer=saver,
            middleware=[
                SummarizationMiddleware(
                    # 摘要模型，在传入fraction时必须配置max_input_tokens
                    model=model,
                    # 触发条件 可以多选 任意一个条件满足就开始总结
                    trigger=[("tokens", 3000), ("messages", 6), ("fraction", 0.001)],
                    # 保留的历史消息
                    keep=("messages", 2),
                    # token计算函数
                    token_counter=count_tokens_approximately,
                    # 给大模型看的提示词，大模型最终输出不一定看这个 主要是把英文转化为中文
                    summary_prompt="上下文会话即将超出模型上下文长度，开始总结\n{messages}",
                    # 生成的摘要的最大token字符
                    trim_tokens_to_summarize="3000",
                )
            ],
        )

        config = {"configurable": {"thread_id": "summary_test"}}

        for i in range(5):
            messages_to_send = [
                HumanMessage(f"这是第{i+1}次提问，问一些旅游相关的问题")
            ]
            response = agent.invoke({"messages": messages_to_send}, config=config)
            rprint(f"第{i+1}次对话:", response["messages"][-1].content[:50], "...")

        state = agent.get_state(config=config)
        all_messages = state.values["messages"]
        rprint(f"\n消息总数: {len(all_messages)}")
        for msg in all_messages:
            rprint(f"  [{msg.type}]: {msg.content[:100]}")


if __name__ == "__main__":
    rprint("=" * 50)
    rprint("消息截断示例")
    rprint("=" * 50)
    truncation_example()

    rprint("\n" + "=" * 50)
    rprint("消息摘要示例")
    rprint("=" * 50)
    summarization_middleware_example()

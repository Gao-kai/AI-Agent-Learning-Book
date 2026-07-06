"""
当我们的Agent需要任务规划和追踪进度的能力时，可以应对负责的多步骤任务。

1. 首先判断你的任务是否需要拆解
2. 再判断步骤是否多变并且需要应对失败的情况
    - 否 直接走langgraph的节点固定
    - 是 用todolist规划
3. 需要在前端UI界面展示Agent的思考和执行进度

Todolist的中间件的本质是调用内置的write_todos工具实现的
每进行一个步骤，agent就会去更新一次todolist
"""

from langchain.agents import create_agent
from langchain.agents.middleware.pii import PIIMiddleware
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_core.messages.human import HumanMessage

from LangChain_Python.models.chat_model import model

# 创建agent
agent = create_agent(
    model=model,
    tools=[],
    middleware=[TodoListMiddleware()],
    system_prompt="你是一个AI代码助手，你需要多步骤任务时先使用todolist的工具制定步骤，然后分步骤实现",
)


# 调用agent
response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                "这是一个复杂的代码解释任务，请你帮我解释下为什么js中0.1+0.2不等于0.3,需要将原理、本质、表现分步骤回答"
            )
        ]
    }
)

# 查看响应
for msg in response["messages"]:
    msg.pretty_print()

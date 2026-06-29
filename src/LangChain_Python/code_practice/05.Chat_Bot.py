import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv(override=True)

# 定义常量
EXIT_KEY_WORD = "exit"
MAX__MESSAGE_PAIRS = 10

# 创建对话模型
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# 创建消息列表
messages: List[BaseMessage] = []

# 创建系统角色
systemMessage = {
    "role": "system",
    "content": "你是一个财经炒股专家，你的名字叫小财迷，需要耐心解答财经理财方面的知识。",
}
messages.append(systemMessage)

# 模拟客服对话
i = 1
while True:
    print("=" * 10, f"当前是第{i}轮对话", "=" * 10)
    user_input = input("请输入问题，小财迷可以帮你耐心解答")

    if user_input == EXIT_KEY_WORD:
        print("=" * 10, f"用户退出会话。", "=" * 10)
        break

    messages.append(HumanMessage(content=user_input))

    response = model.stream(messages)

    chunk_message = ""
    for chunk in response:
        print(chunk.content, end="", flush=True)
        chunk_message += chunk.content

    messages.append(AIMessage(content=chunk_message))

    i += 1

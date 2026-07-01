from typing import List
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich import print as rich_print
import os

"""
LangChain提供的ChatPromptTemplate类相比于字符串方式构建提示词模版来说，具有以下优点：
1. 结构清晰，通过变量占位
2. 易维护和复用，可以单独将提示词模版抽取为单独文件进行管理
3. 自动变量校验
4. 支持多轮对话、RAG、Few-Shot等复杂场景
5. 便于调试和日志追踪


# 1. 如何初始化提示词模版？

1. 直接调用ChatPromptTemplate，传入提示词模版
2. 调用ChatPromptTemplate.from_messages方法，只有字典和元组列表两种方式支持变量占位，BaseMessage对象列表不会处理变量占位
返回的是一个ChatPromptTemplate的实例

# 方式1: 使用 BaseMessage 对象列表 注意通过这种方法不会处理变量占位
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个烹饪专家"),
    HumanMessage(content="请告诉我湖南小炒肉的做法"),
])

# 方式2: 使用字典列表（OpenAI 风格）
prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "你是一个烹饪专家"},
    {"role": "user", "content": "请告诉我湖南小炒肉的做法"},
])

# 方式3: 使用元组列表
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个烹饪专家"),
    ("user", "请告诉我湖南小炒肉的做法"),
])

# 2. 如何调用提示词模版实例
1. invoke方法 传入变量的对象 返回PromptValue实例
2. format方法 传入变量，返回字符串
3. format_message 传入变量，返回消息列表

以上三个方法返回的值都可以直接通过model.invoke进行调用
"""


# 加载环境变量 override表示强制覆盖之前的相同Key的环境变量
load_dotenv(override=True)

# 从环境变量获取环境变量
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化Model
model = init_chat_model(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

# 初始化ChatPromptTemplate实例
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "假设你是一个{type}专家，你的风格是简单有条理的告诉用户的问题"),
        ("user", "你好"),
        ("ai", "你好，请问你要咨询什么问题？"),
        ("user", "{user_input}"),
    ]
)

# 传入变量 构建提示词模版
chat_prompt_value = chat_prompt_template.invoke(
    {"type": "厨师", "user_input": "我想做一道小炒黄牛肉"}
)
print(chat_prompt_value)
print(type(chat_prompt_value))

# 模型调用提示词模版
response = model.invoke(chat_prompt_value)
rich_print(response.content)

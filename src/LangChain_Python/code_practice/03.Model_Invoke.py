"""
基于Model Invoke来进行模型调用

模型可以通过invoke方法来进行调用，支持传入三种类型的参数：
- 字符串：直接传入模型调用的字符串
- 消息对象列表：传入一个包含多个消息对象的列表，每个消息对象对应一个消息
    - 消息对象AIMessage，其底层是基于对role等于"assistant"的消息对象
    - 消息对象HumanMessage，其底层是基于对role等于"user"的消息对象
    - 消息对象SystemMessage，其底层是基于对role等于"system"的消息对象
- 字典：传入一个包含多个消息的字典，每个消息包含role和content两个键
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 加载环境变量 override表示强制覆盖之前的相同Key的环境变量
load_dotenv(override=True)

# 从环境变量获取环境变量
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化Model
model = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, base_url=BASE_URL)

# 01 - 基于字典进行模型调用
messages = [
    {"role": "system", "content": "你是一个烹饪专家"},
    {"role": "user", "content": "请告诉我湖南小炒肉的具体做法，分步骤回答"},
]
result = model.invoke(messages)
print(result.content)


# 02 - 基于消息对象列表进行模型调用
messages.append(AIMessage(content=result.content))
messages.append(HumanMessage(content="我刚才问了什么问题？"))
result = model.invoke(messages)
print(result.content)


from rich import print as rich_print
rich_print(result)

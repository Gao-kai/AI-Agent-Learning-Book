"""
Content Block
1. 在输入层面，比传统的content字段在构建消息列表的时候更加类型化，并且支持多模态输入
2. 在输出层面，更加注重对于extra_body字段的处理，支持自定义的思考推理过程
"""

from typing import List
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich import print as rich_print
import os

# 加载环境变量 override表示强制覆盖之前的相同Key的环境变量
load_dotenv(override=True)

# 从环境变量获取环境变量
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化Model 开启思考推理过程
model = init_chat_model(
    model_provider="deepseek",
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
    extra_body={"thinking": {"type": "enabled"}},
)

# 使用content_blocks参数更加精确描述消息内容 且支持多模态输入
messages: List[BaseMessage] = [
    SystemMessage(
        content_blocks=[
            {"type": "text", "text": "你是一个烹饪专家"},
            # {"type": "image", "image_url": "https://example.com/image.jpg"},
        ]
    ),
    HumanMessage(
        content_blocks=[
            {"type": "text", "text": "请告诉我湖南小炒肉的具体做法，分步骤回答"},
        ],
    ),
]
result = model.invoke(messages)
rich_print(result.content)  # 只返回回答
rich_print(result.content_blocks)  # 分别返回type=reasoning推理过程和type=text的回答

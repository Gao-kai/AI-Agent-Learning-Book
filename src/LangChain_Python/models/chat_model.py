from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

"""
基于LangChain统一封装的API init_chat_model实现
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
    extra_body={"thinking": {"type": "disabled"}},
)

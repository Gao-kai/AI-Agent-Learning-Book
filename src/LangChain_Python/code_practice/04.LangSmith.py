from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from rich import print as rich_print

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
    configurable_fields="any",
)

# 定义模型调用时额外参数配置
config: RunnableConfig = {
    "run_name": "langSmith-demo-01",
    "tags": ["langchain", "test", "booker"],
    "metadata": {"user_id": "001", "session_id": "002"},
    "max_concurrency": 10,
    "configurable": {"model": "deepseek-v4-flash"},
}

# 基于字典进行模型调用
messages = [
    {"role": "system", "content": "你是一个烹饪专家"},
    {"role": "user", "content": "请告诉我湖南小炒肉的具体做法，分步骤回答"},
]
result = model.invoke(messages, config=config)
print(result.content)
rich_print(result)

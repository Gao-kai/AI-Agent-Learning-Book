from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

"""
基于Model Class来进行模型调用

LangChain为常见的大模型厂商都提供了各个厂商自定义的API，比如ChatOpenAi，ChatDeepSeek等
由于ChatOpenAi类集成大部分模型厂商的API规范，因此我们使用ChatOpenAI就可以满足开发要求

常见的模型参数如下：
- model：模型名称，比如gpt-3.5-turbo
- api_key：OpenAI API密钥
- base_url：OpenAI API基础URL
- temperature：温度参数，控制模型输出的随机性
- max_tokens：最大输出token数
- top_p：top_p参数，控制模型输出的随机性
"""


# 加载环境变量 override表示强制覆盖之前的相同Key的环境变量
load_dotenv(override=True)

# 从环境变量获取环境变量
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化Model
model = ChatOpenAI(
    model=MODEL_NAME, 
    api_key=OPENAI_API_KEY, 
    base_url=BASE_URL,
)



# 模型调用
result = model.invoke("请你介绍下你自己",)
print(result)

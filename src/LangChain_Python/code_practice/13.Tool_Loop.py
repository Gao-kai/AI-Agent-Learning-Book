"""
多工具循环调用案例

1. 使用LLM需要我们手动处理多工具循环调用 直到没有工具可以调用的时候结束对话
2. Agent可以自动处理工具循环调用
"""

import os

from dotenv import load_dotenv
from langchain_core.messages.human import HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from rich import print as rprint
from tenacity import retry, stop_after_attempt

from src.LangChain_Python.models.chat_model import model

load_dotenv(override=True)


# 定义天气对象
class WeatherInput(BaseModel):
    city: str = Field(default="Beijing", description="城市名称")
    is_china: bool = Field(default=True, description="是否在中国境内")
    query_type: Literal["摄氏度", "华氏度"] = Field(
        description="温度类型，华氏度或者摄氏度", default="摄氏度"
    )


@tool(args_schema=WeatherInput, description="查询中国某地的天气")
def get_weather(city: str, is_china: bool, query_type: Literal["摄氏度", "华氏度"]):
    """
    查询某地的天气
    """
    return f"当前{city}气温是{query_type}32度，位于中国境内，"


# 定义美食对象
class FoodInput(BaseModel):
    city: str = Field(default="Beijing", description="城市名称")


@retry(stop=stop_after_attempt(3))
@tool(args_schema=FoodInput, description="查询中国某地的美食")
def get_food(city: str):
    """
    查询某地的美食
    """
    return f"当前{city}的美食有饺子和火锅"


# 初始化Model
# model = ChatOpenAI(
#     model=os.getenv("MODEL_NAME"),
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("BASE_URL"),
# )

messages = [
    HumanMessage(content="帮我查下北京的天气，顺带告诉我北京有什么美食可以推荐？")
]

model_bind_tools = model.bind_tools([get_food, get_weather])

# 工具循环调用
while True:
    response = model_bind_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        break

    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_food":
            food_tool_result = get_food.invoke(tool_call)
            messages.append(food_tool_result)
        if tool_call["name"] == "get_weather":
            get_weather_result = get_weather.invoke(tool_call)
            messages.append(get_weather_result)

# 循环调用结束 输出最终答案
rprint(response)

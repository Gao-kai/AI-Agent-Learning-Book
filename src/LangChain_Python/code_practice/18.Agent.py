"""
本章节主要阐述：Agent的核心概念

一个Agent必须具有以下四个能力：
1. Action行动：可以执行具体的操作，比如调用外部API、操作文件、返回响应
2. Tools工具：Agent应该可以调用外部的工具，来实现负责的任务
3. Memory记忆：Agent应该具有短期记忆和长期记忆来存储历史上下文信息
4. Planning规划决策能力：Agent应该具有反思、自我批评、思维链和子目标分解的能力

除此之外，Agent之间应该是可以协作的，比如一个Agent可以调用另一个Agent的Action行动，或者一个Agent可以调用另一个Agent的Tools工具。

==============================================================================

LangChain中如何快速搭建一个Agent
- 模型传入:支持字符串/模型实例调用
- Agent的调用，注意传入的是一个包含messages的字典
- Agent的输出，返回的是消息列表
- Agent绑定内置工具和外部工具
"""

import os
from typing import Literal

from langchain.agents import create_agent
from IPython.display import Image, display
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from src.LangChain_Python.models.chat_model import model
from rich import print as rprint


# 定义Schema对象
class WeatherInput(BaseModel):
    city: str = Field(default="Beijing", description="城市名称")
    query_type: Literal["摄氏度", "华氏度"] = Field(
        description="温度类型，华氏度或者摄氏度", default="摄氏度"
    )


# 创建自定义工具
@tool(args_schema=WeatherInput, description="查询中国某地的天气")
def get_weather(city: str, query_type: Literal["摄氏度", "华氏度"]):
    """
    查询某地的天气
    """
    return f"当前{city}气温是{query_type}32度，位于中国境内，"


# 创建内置工具
tavily_search_tool = TavilySearch(
    max_results=5, topic="general", tavily_api_key=os.getenv("TAVILY_API_KEY")
)

# 创建Agent绑定工具
agent = create_agent(
    model=model,
    tools=[tavily_search_tool, get_weather],
    system_prompt="假设你是一名智能的AI旅游助手",
)

# 查看Agent的图表示（agent的本质是一个编译好的StateGraph）
graph = agent.get_graph().draw_mermaid_png()
display(Image(graph))


# 调用Agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我查询下北京这座城市的著名景点，然后顺便查询下北京的天气",
            },
        ]
    }
)


rprint(response)

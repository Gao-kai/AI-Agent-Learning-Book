"""
Agent的异常处理

我们在使用ToolStrategy的时候，可以传入三个参数来进一步细化结构化输出的场景：
1. schema 结构化输出的模型 也就是Pydantic Json Schema dataclass TypedDict等数据类型
2. tool_message_content 结构化输出返回的ToolMessage消息中的content占位符 可以节省token
3. handle_errors 数据校验失败时的重试策略 默认为True
    - True 使用内置的错误处理异常来提示模型重试，推荐配置
    - False 不处理直接报错，不重试
    - 字符串 捕获异常后使用该字符串来当作错误的消息
    - 指定类型ExceptionType 只对特定类型异常进行重试
    - 函数 高度自定义错误异常处理
"""

import os
from pyexpat.errors import messages
from typing import Literal, Union
from langchain.agents import create_agent
from IPython.display import Image, display
from langchain.agents.structured_output import ToolStrategy
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


# 创建结构化输出模型
class PersonSchema(BaseModel):
    name: str = Field(description="名称")
    age: str = Field(description="年龄")
    hobby: str = Field(description="爱好")


class AnimalSchema(BaseModel):
    name: str = Field(description="动物名称")
    type: str = Field(description="类型，爬行动物或飞行动物")
    section: str = Field(description="分布区域")


# 创建Agent绑定工具
agent = create_agent(
    model=model,
    tools=[tavily_search_tool, get_weather],
    name="travel_agent",
    system_prompt="""
    角色：你是一名智能的AI助手助手

    工具：查询天气时用get_weather，网络搜索用tavily_search_tool

    重试机制：如果某次工具调用返回不可用或者报错，可以尝试三次，三次之后不再尝试，抛出异常
    """,
    response_format=ToolStrategy(
        schema=Union[PersonSchema, AnimalSchema],
        tool_message_content="返回响应已经成功转化为结构化输出",
        handle_errors=True,
    ),
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
                "content": "帮我查询提取信息为结构化输出：小明13岁了爱好是打篮球，小狗快乐的跑着，它是一只中华田园犬",
            },
        ]
    },
)
# rprint(response)
print(response["structured_response"])

for msg in response["messages"]:
    msg.pretty_print()

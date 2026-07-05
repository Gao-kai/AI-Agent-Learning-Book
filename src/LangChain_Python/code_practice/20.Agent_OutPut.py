"""
Agent结构化输出

1. LLM Model通过调用with_structure_output来实现，Agent通过response_format时调用
2. response_format支持多种结构化策略：
    - None 自然语言，不进行格式化
    - Auto模式：直接传入一个类型Schema会包装为Auto，触发自动选择策略，如果模型支持原生支持比如OpenAI、Claude等就走Provider；否则就走Tool
    - Tool模式：将结构化输出的任务包装为一个虚拟工具，在获取到最终输出之前调用这个工具进行格式化（推荐✅ 适配大多数模型）
    - Provider：原生支持比如OpenAI、Claude等就走Provider
3. 从Agent返回的response的structured_response字段中获取结构化输出，注意模型如果时flash等轻量级的可能不会返回这个字段
4. tool_message_content设置末尾插入的最后一条ToolMessage的content，可以节省Token
"""

import os
from pyexpat.errors import messages
from typing import Literal
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
class AgentOutputSchema(BaseModel):
    question: str = Field(description="用户问题")
    query_date: str = Field(description="查询时间")
    answer: str = Field(description="用户答案")


# 创建Agent绑定工具
agent = create_agent(
    model=model,
    tools=[tavily_search_tool, get_weather],
    name="travel_agent",
    system_prompt="""
    角色：你是一名智能的AI旅游助手

    工具：查询天气时用get_weather，网络搜索用tavily_search_tool

    重试机制：如果某次工具调用返回不可用或者报错，可以尝试三次，三次之后不再尝试，抛出异常
    """,
    response_format=ToolStrategy(
        schema=AgentOutputSchema,
        tool_message_content="返回响应已经成功转化为结构化输出",
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
                "content": "帮我查询下北京这座城市的著名景点，然后顺便查询下北京的天气",
            },
        ]
    },
    # config={"recursion_limit": 10},
)
# rprint(response)
print(response["structured_response"])
for msg in response["messages"]:
    msg.pretty_print()

"""
Agent高级

1.Agent的Re-Act模式：不断的通过对任务进行思考推理-执行工具-再思考-再执行，直到没有工具可供调用后Agent返回最终结果
2.Agent在create_agent时的系统提示词配置(为Agent提供任务背景、行为准则和操作指南，明确角色、输出、工具调用时机等)
3.create_agent时设置recurision_limit可以限制工具调用最大次数
4.create_agent时设置name可以设置Agent的名称，这在多Agent结构中非常有用，用来做Agent的身份标记，方便后期调试和子图节点的接入
"""

import os
from pyexpat.errors import messages
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
    name="travel_agent",
    system_prompt="""
    角色：你是一名智能的AI旅游助手
    
    工具：查询天气时用get_weather，网络搜索用tavily_search_tool
    
    重试机制：如果某次工具调用返回不可用或者报错，可以尝试三次，三次之后不再尝试，抛出异常
    """,
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
    config={"recursion_limit": 5},
)


for msg in response["messages"]:
    msg.pretty_print()

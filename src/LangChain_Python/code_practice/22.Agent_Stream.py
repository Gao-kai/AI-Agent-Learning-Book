"""
Agent流式输出与结构化数据获取

1. 使用 stream_events API（官方推荐）
   - version="v3" 支持分别消费 messages、tool_calls、values 等事件
   - 通过 stream.interleave() 同时监听多种事件类型
   - 通过 stream.output 获取最终状态（包含结构化输出）

2. 流式输出中获取结构化数据的方法：
   - 方法1：使用 stream_events API，在最后通过 stream.output 获取
   - 方法2：使用 stream_mode=["messages", "updates"]，在 updates 中查找

3. 注意：stream_mode="messages" 只返回文本Token，无法获取结构化数据
"""

import os
from typing import Literal
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from src.LangChain_Python.models.chat_model import model


# 定义Schema对象
class WeatherInput(BaseModel):
    city: str = Field(default="Beijing", description="城市名称")
    query_type: Literal["摄氏度", "华氏度"] = Field(
        description="温度类型，华氏度或者摄氏度", default="摄氏度"
    )


# 创建自定义工具
@tool(args_schema=WeatherInput, description="查询中国某地的天气")
def get_weather(city: str, query_type: Literal["摄氏度", "华氏度"]):
    """查询某地的天气"""
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

        输出要求：
        - 完成任务后，必须使用结构化输出工具返回结果
        - 结构化输出需要包含：question（用户问题）、query_date（查询时间）、answer（用户答案）
        - answer字段应该包含详细的景点介绍和天气信息

        重试机制：如果某次工具调用返回不可用或者报错，可以尝试三次，三次之后不再尝试，抛出异常
    """,
    response_format=AgentOutputSchema,
)


# 方法2：使用 stream 方法配合 stream_mode=["messages", "updates"]
print("\n\n" + "=" * 50)
print("=== 方法2：使用 stream 方法 ===")
stream = agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "帮我查询上海的著名景点，每个景点的介绍最多100字符即可，最多三个景点即可",
            },
        ]
    },
    stream_mode=["messages", "updates"],
    version="v2",
)


for chunk in stream:
    # 1. 处理实时 token（思考过程、工具调用片段）
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if hasattr(token, "text") and token.text:
            print(token.text, end="", flush=True)

    # 2. 处理状态更新（从末尾获取最终结构化结果）
    elif chunk["type"] == "updates":
        for node, data in chunk["data"].items():
            # 检查是否包含结构化响应
            if "structured_response" in data:
                print("\n--- 最终结构化结果 ---")
                print(data["structured_response"])
                final_output = data["structured_response"]

print(f"结构化输出：{final_output}")

"""
大模型不能主动调用Tool
Agent可以主动调用Tool
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import ToolMessage, AIMessage
from langchain.tools import tool
from models.chat_model import model


# 1. @tool 定义查询天气工具
@tool
def get_weather(location: str) -> str:
    """查询天气"""
    return f"查询{location}的天气是23摄氏度"


# 2. 创建历史消息
messages = [
    {"role": "system", "content": "你是一个通用AI助手"},
    {"role": "user", "content": "请告诉我湖南长沙的天气"},
]

# 3. 工具调用
model = model.bind_tools([get_weather])
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

# 4. 检查是否有工具调用
if result.tool_calls:
    for tool_call in result.tool_calls:
        if tool_call["name"] == "get_weather":
            tool_call_result = get_weather.invoke(tool_call)
            print(tool_call_result)
            messages.append(tool_call_result)
        else:
            print("其他工具调用")

# 5. 最后将工具调用结果一起给大模型
response = model.invoke(messages)
print(f"大模型最终回复：{response}")

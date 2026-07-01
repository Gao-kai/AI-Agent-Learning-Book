"""
LangChain中支持两种方式定义工具：
1. 使用官方提供的@tool装饰器
    - parse_docstring=True开启注释的解析，但是一旦开启如果出现语法错误就会报错
2. 自己定义不加装饰器打，但是需要写docstring说明
    - 工具描述 必须
    - 参数类型
    - 返回值
    - 参数的默认值

两种方法底层调用的都是convert_to_openai_tool方法,最终会生成pydantic模式的工具描述，然后转换成规范的tool_schema格式：
{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': '查询某地在某一天的天气',
        'parameters': {
            'properties': {
                'date_str': {'description': '日期', 'type': 'string'},
                'city': {
                    'default': '北京',
                    'description': '城市名称',
                    'type': 'string'
                }
            },
            'required': ['date_str'],
            'type': 'object'
        }
    }
}

"""

import os
from typing import Dict

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv(override=True)


@tool(parse_docstring=True)
def get_user(user_id: str) -> Dict[str, str]:
    """
    查询用户的详细信息

    Args:
        user_id: 用户ID

    Returns:
        返回查询的结果
    """
    return {"name": "zhangsan"}


def get_weather(date_str: str, city: str = "北京"):
    """
    查询某地在某一天的天气

    Args:
        date_str: 日期
        city: 城市名称

    Returns:
        返回天气查询的结果
    """
    return f"{city}在{date_str}这一天的气温是32摄氏度。"


# 初始化Model
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

rprint(convert_to_openai_tool(get_weather))

# 绑定自定义工具
model_with_tools = model.bind_tools([get_weather])

# 调用工具
model_with_tools.invoke("帮我查询上海的天气")

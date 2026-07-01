"""
在LangChain中定义工具中参数的类型有两种方案：
1. Json Schema：动态生成，可以集成数据库配置或者用户输入来运行时动态生成
2. pydantic方案：静态生成
"""

import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from rich import print as rprint

load_dotenv(override=True)


# 定义对象
class WeatherInput(BaseModel):
    city: str = Field(default="Beijing", description="城市名称")
    is_china: bool = Field(default=True, description="是否在中国境内")
    query_type: Literal["摄氏度", "华氏度"] = Field(
        description="温度类型，华氏度或者摄氏度", default="摄氏度"
    )


"""
基于面向对象的Pydantic方式实现类型校验
1. 定义对象WeatherInput
2. 通过Field来约束类型，描述和默认值
3. 通过@tool(args_schema=WeatherInput, description="查询中国某地的天气")开启类型校验
4. 将下面这个对象的parameters传递可以当作json schema传递给tool 也可以实现同样的效果
最终会转化为这种形式：
{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': '查询中国某地的天气',
        'parameters': {
            'properties': {
                'city': {
                    'default': 'Beijing',
                    'description': '城市名称',
                    'type': 'string'
                },
                'is_china': {
                    'default': True,
                    'description': '是否在中国境内',
                    'type': 'boolean'
                },
                'query_type': {
                    'default': '摄氏度',
                    'description': '温度类型，华氏度或者摄氏度',
                    'enum': ['摄氏度', '华氏度'],
                    'type': 'string'
                }
            },
            'type': 'object'
        }
    }
}

"""


@tool(args_schema=WeatherInput, description="查询中国某地的天气")
def get_weather(city: str, is_china: bool, query_type: Literal["摄氏度", "华氏度"]):
    """
    查询某地的天气
    """
    return f"当前{city}气温是{query_type}32度，位于中国境内，"


rprint(convert_to_openai_tool(get_weather))

# 初始化Model
model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

"""
# LangChain提供的with-structured_output方法优点

    1. 不用在提示词中写以Json格式返回，字段为....
    2. 不需要对大模型返回的Json文本字符串结果手动再进行解析
    3. 不需要手动验证类型是否合格

# with_structured_output的底层工作原理

    1.定义Pydantic模型类
    2.Langchain在运行时自动调用底层的json_schema方法转化为标准的JSON Schema
    3.model在调用with_structured_output时通过工具调用的方法将标准的JSON Schema传递给大模型
    4.大模型返回结构化Json字符串
    5.将JSON字符串转化为字典对象
    6.将字典对象转化为Pydantic对象的实例，并进行类型校验
    7.如果校验通过，则返回Pydantic对象的实例

# with_structured_output方法参数
    1. 第一个参数是输出的模式，支持：
        - 定义的Pydantic对象类（注意只有它返回的是Schema实例 并且支持类型校验 描述 嵌套结构等 推荐）
        - Json schema
        - TypedDict
    2. 第二个参数可以设置include_raw=True参数，表示返回解析前的原始的AIMessage，如果不传则只返回解析后的JSON对象
    3. 模型调用后的返回值直接就是定义的Pydantic对象实例，不需要再进行转换

# Pydantic的用法
    1. 所有结构化输出的数据模型必须继承BaseModel
    2. 基于Field实现描述和默认值等配置
    3. 基于Optional实现可选参数
    4. 基于Enum枚举类

"""

from enum import Enum
from typing import Optional

from langchain_core.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.LangChain_Python.models.chat_model import model


# 定义枚举类型
class LevelEnum(str, Enum):
    High = "好看"
    Middle = "一般"
    Low = "烂片"


# 定义Pydantic Schema对象（所有结构化输出的数据模型必须继承BaseModel）
class MovieSchema(BaseModel):
    name: str = Field(default="", description="电影名称")
    date: Optional[str] = Field(description="上映日期")
    author: LevelEnum = Field(description="电影评价")


# 定义提示词模版
template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个电影专家（你的回答需要用JSON格式返回）"),
        ("user", "请问电影《{user_input}》怎么样? "),
    ]
)

prompt = template.invoke({"user_input": "霸王别姬"})

model_with_structure = model.with_structured_output(MovieSchema)
response = model_with_structure.invoke(prompt)
print(f"电影名称：{response.name}")
print(f"上映日期：{response.date}")
print(f"电影评价：{response.author}")

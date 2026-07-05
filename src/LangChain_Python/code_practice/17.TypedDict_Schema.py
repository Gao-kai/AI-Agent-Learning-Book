from enum import Enum
from typing import List, TypedDict, Annotated
from rich import print as rprint
from pydantic import BaseModel, Field
from src.LangChain_Python.models.chat_model import model

"""
本章节主要介绍如何通过Python语言自带的TypedDict和Annotated来实现结构化输出
主要注意：
1. 通过TypedDict返回的大模型响应是<class 'dict'>，而Pydantic返回的是一个类的实例
2. 需要配合Annotated来进行使用，包含字段的类型、描述和是否强制要求生成的结构包含此字段
3. 只有通过pydantic方式的schema构建的大模型，在大模型字段返回的不符合定义时才会报错，其他比如TypedDict、Json schema、@dataclass都不会报错
"""


# 定义枚举类
class GenderEnum(str, Enum):
    man = "男"
    women = "女"


# 定义演员Schema
class Actor(TypedDict):
    name: Annotated[str, ..., "演员名称"]
    gender: Annotated[GenderEnum, ..., "演员性别"]


# 定义Movie Schema
class MovieSchema(TypedDict):
    name: Annotated[str, ..., "电影的名称，不能超出40个字符"]
    actor_list: Annotated[List[Actor], ..., "演员列表"]
    comment: Annotated[str, ..., "电影评价，最小必须100个字符"]


# 模型调用
model_with_structure = model.with_structured_output(MovieSchema)
response = model_with_structure.invoke("请你给我《追凶者也》这部电影的介绍")
print(type(response))
rprint(response)

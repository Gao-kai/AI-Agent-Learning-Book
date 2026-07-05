"""
本章节展示Pydantic的高级用法
1. 列表提取
2. 条件限制
3. 结构嵌套（最多不要超出3层并且应该有清晰的描述信息）
4. 原理

## 模型调用后结构化输出如下信息：
Movie(
    name='霸王别姬',
    actor_list=[
        Actor(name='张国荣', gender=<GenderEnum.man: '男'>),
        Actor(name='张丰毅', gender=<GenderEnum.man: '男'>),
        Actor(name='巩俐', gender=<GenderEnum.women: '女'>),
        Actor(name='葛优', gender=<GenderEnum.man: '男'>)
    ],
    comment='《霸王别姬》是陈凯歌导演的经典之作，改编自李碧华的同名小说。影片以
京剧为背景，讲述了程蝶衣（张国荣饰）和段小楼（张丰毅饰）两位京剧艺人跨越半个世纪
的悲欢离合。程蝶衣饰演的虞姬与段小楼饰演的霸王，在戏台上风华绝代，但在现实生活中
却因时代变迁、情感纠葛而命运多舛。影片深刻展现了人性的复杂、艺术的执着以及历史洪
流下个体的无奈与挣扎，是中国电影史上不可多得的巅峰之作，曾获得戛纳电影节金棕榈奖
。'
)
"""

from enum import Enum
from typing import List
from rich import print as rprint
from pydantic import BaseModel, Field
from src.LangChain_Python.models.chat_model import model


# 定义枚举类
class GenderEnum(str, Enum):
    man = "男"
    women = "女"


# 定义演员Schema
class Actor(BaseModel):
    name: str = Field(description="演员名称")
    gender: GenderEnum = Field(description="演员性别")


# 定义电影Schema
class Movie(BaseModel):
    # max_length 约束输出最大长度
    name: str = Field(description="电影名称", max_length=40)

    # List[Actor]实现结构嵌套
    actor_list: List[Actor] = Field(description="演员列表")

    # min_length 约束最小输出长度
    comment: str = Field(description="电影评价", min_length=100)


# 模型调用
model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("霸王别姬这部电影介绍")
rprint(response)

"""
本文件主要阐述LangChain中工具调用的最佳实践：
1. 每一个工具只做一件事，符合编程最基本的单一原则
2. 对于工具的描述要清楚，职责范围要清晰，这样子模型在调用时才会更加准确
   如果使用docstring的方式就要严格按照规范来进行描述、参数类型和返回值的定义
3. 增加重试机制
    - 工具函数级别：工具函数中需要捕获异常然后处理为字符串返回，不要直接抛出错误
    - Agent级别：需要在create agent的时候强调如果工具失败尝试使用其他方式解决问题的兜底
    - 模型调用级别: @retry(stop=stop_after_attempt(3))尝试3次后再报错
4. 工具的返回尽量是字符串纯文本，不要返回太复杂的字典或者数据结构，因为llm本身就是处理文本的
5. 在I/O密集型任务比如API调用、数据库操作、文件操作中使用异步函数或许是最佳选择
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt


class FoodInput(BaseModel):
    city: str = Field(default="Beijing", description="城市名称")


@retry(stop=stop_after_attempt(3))
@tool(args_schema=FoodInput, description="查询中国某地的美食")
def get_food(city: str):
    """
    查询某地的美食
    """
    return f"当前{city}的美食有饺子和火锅"

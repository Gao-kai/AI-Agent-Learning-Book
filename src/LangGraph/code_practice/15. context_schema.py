"""
Runtime Context

1. 将不属于全局状态的数据可以放在一个context上下文中去访问
2. 将运行时配置和图的状态分离，互不干扰
3. 非常适合保存API密钥、数据库配置信息等上下文信息

如何配置：
1. 基于@dataclass来定义ContextSchema类型
2. 创建图的时候builder = StateGraph(OverallState, context_schema=ContextSchema)通过第二个参数指定类型
3. 图调用执行的时候传入context参数 graph.invoke(initial_state, context=context)
3. 节点函数中通过第二个参数runtime接受context.runtime中的参数
"""

from dataclasses import dataclass
from typing import TypedDict, Annotated, List
import operator

from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END


# 定义图状态
class OverallState(TypedDict):
    messages: Annotated[List[str], operator.add]
    response: str


# 定义上下文Context 基于装饰器dataclass
@dataclass
class ContextSchema:
    model_name: str
    db_connection: str
    api_key: str


# 在节点中使用Context信息 基于第二个参数runtime
def process_message(state: OverallState, runtime: Runtime[ContextSchema]):
    print("处理用户数据")
    model_name = runtime.context.model_name
    return {"message": ["用户数据处理完成"], "response": model_name}


# 创建图，指定 state_schema 和 context_schema
builder = StateGraph(OverallState, context_schema=ContextSchema)

# 添加节点
builder.add_node("process_message", process_message)

# 添加边
builder.add_edge(START, "process_message")
builder.add_edge("process_message", END)

# 编译图
graph = builder.compile()

# 定义初始状态
initial_state = {
    "messages": ["请帮我查询最新的订单信息"],
    "response": "",
}

# 定义上下文
context = ContextSchema(
    model_name="gpt-4-turbo",
    db_connection="postgresql://user:pass@localhost:5432/orders_db",
    api_key="sk-abcdefghijklmnopqrstuvwxyz123456",
)

graph.invoke(initial_state, context=context)

from xxlimited_35 import Null

from langgraph.graph.state import StateGraph
from langgraph.graph import END, START
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Annotated, Dict
import operator

"""
自定义reducer函数
1. 字典里key存在，则新的值覆盖旧的值
2. 字典里key不存在，则依然保留这个键，并设置值为null
"""


def custom_reducer(
    curr_value: Dict[str, any], new_value: Dict[str, any]
) -> Dict[str, any]:
    curr_value_copy = curr_value.copy()
    for key, value in new_value.items():
        if key in curr_value_copy:
            curr_value_copy[key] = value
        else:
            curr_value_copy[key] = Null
    return curr_value_copy


# 定义全局状态
class OverallState(TypedDict):
    messages: Annotated[List, add_messages]
    userList: Annotated[List[str], operator.add]
    count: Annotated[int, operator.add]
    metadata: Annotated[Dict[str, any], custom_reducer]


# 定义消息节点
def chat_node(state: OverallState) -> OverallState:
    return {
        "messages": [("ai", "Hello 我有什么可以帮你呢？")],
        "userList": ["lily"],
        "count": 1,
        "metadata": {"id": 100},
    }


# 定义用户节点
def user_node(state: OverallState) -> OverallState:
    return {
        "messages": [("user", "帮我查询今天北京的天气")],
        "userList": ["Tom"],
        "count": 1,
        "metadata": {"id": 100, "stars": 990},
    }


# 构建图
graph_builder = StateGraph(OverallState)
graph_builder.add_node("chat_node", chat_node)
graph_builder.add_node("user_node", user_node)
graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", "user_node")
graph_builder.add_edge("user_node", END)
graph = graph_builder.compile()
result = graph.invoke({"metadata": {"id": 99}})
print(f"最终输出结果为：{result}")

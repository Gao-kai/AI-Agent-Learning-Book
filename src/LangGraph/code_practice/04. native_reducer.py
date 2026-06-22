from langgraph.graph.state import StateGraph
from langgraph.graph import END, START
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Annotated
import operator


# 定义全局状态
class OverallState(TypedDict):
    messages: Annotated[List, add_messages]
    userList: Annotated[List[str], operator.add]
    count: Annotated[int, operator.add]


# 定义消息节点
def chat_node(state: OverallState) -> OverallState:
    return {
        "messages": [("ai", "Hello 我有什么可以帮你呢？")],
        "userList": ["lily"],
        "count": 1,
    }


# 定义用户节点
def user_node(state: OverallState) -> OverallState:
    return {
        "messages": [("user", "帮我查询今天北京的天气")],
        "userList": ["Tom"],
        "count": 1,
    }


# 构建图
graph_builder = StateGraph(OverallState)
graph_builder.add_node("chat_node", chat_node)
graph_builder.add_node("user_node", user_node)
graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", "user_node")
graph_builder.add_edge("user_node", END)
graph = graph_builder.compile()
result = graph.invoke({"messages": []})
print(f"最终输出结果为：{result}")

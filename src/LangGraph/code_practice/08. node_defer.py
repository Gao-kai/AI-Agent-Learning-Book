"""
LangGraph中支持设置将某个节点的执行推迟到所有其他待处理任务完成后执行
场景：某个节点的执行必须等待其他所有并行分支任务完成之后才可以开始执行

"""

import operator
import time
from typing import TypedDict, Annotated, List

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import START, END, StateGraph
from langgraph.types import CachePolicy


class OverallState(TypedDict):
    pathList: Annotated[List[str], operator.add]


# 节点A
def node_a(state: OverallState) -> OverallState:
    print(f"进入节点A")
    return {"pathList": ["A"]}


# 节点B
def node_b(state: OverallState) -> OverallState:
    print(f"进入节点B")
    return {"pathList": ["B"]}


# 节点B-2
def node_b_2(state: OverallState) -> OverallState:
    print(f"进入节点B-2")
    return {"pathList": ["B-2"]}


# 节点B-3
def node_b_3(state: OverallState) -> OverallState:
    print(f"进入节点B-3")
    return {"pathList": ["B-3"]}


# 节点C
def node_c(state: OverallState) -> OverallState:
    print(f"进入节点C")
    return {"pathList": ["C"]}


# 节点D
def node_d(state: OverallState) -> OverallState:
    print(f"进入节点D")
    return {"pathList": ["D"]}


# 构建图
graph_builder = StateGraph(OverallState)

# 添加节点
graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)
graph_builder.add_node("node_b_2", node_b_2)
graph_builder.add_node("node_b_3", node_b_3)
graph_builder.add_node("node_c", node_c)

# 设置节点D为Defer节点 必须等到其他并行分支都完成任务才开始执行
graph_builder.add_node("node_d", node_d, defer=True)


# 构建边
graph_builder.add_edge(START, "node_a")
graph_builder.add_edge("node_a", "node_b")
graph_builder.add_edge("node_a", "node_c")
graph_builder.add_edge("node_b", "node_b_2")
graph_builder.add_edge("node_b_2", "node_b_3")
graph_builder.add_edge("node_b_3", "node_d")
graph_builder.add_edge("node_c", "node_d")
graph_builder.add_edge("node_d", END)


# 编译图
graph = graph_builder.compile()

print(f"开始执行工作流")
print(f"计算的结果为：{graph.invoke({"pathList": []})}")

"""
开始执行工作流
进入节点A
进入节点B
进入节点C => 本来C执行
进入节点B-2
进入节点B-3
进入节点D
计算的结果为：{'pathList': ['A', 'B', 'C', 'B-2', 'B-3', 'D']}
"""

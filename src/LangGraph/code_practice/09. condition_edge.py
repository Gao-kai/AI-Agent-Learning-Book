import os
from typing import Literal, Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator


def generate_mermaid_images(graph):
    current_filename = os.path.splitext(os.path.basename(__file__))[0]
    draw_mermaid_path = os.path.join(
        os.path.dirname(__file__), "../mermaid_images", f"{current_filename}.png"
    )
    graph.get_graph().draw_mermaid_png(output_file_path=draw_mermaid_path)


"""
条件节点 add_conditional_edges

source str 条件分支的起点节点名称 
route_condition callable 决定走哪条路径的函数 
path_map dict 路径映射表（返回值 → 目标节点）
"""


# 定义状态
class OverallState(TypedDict):
    value: Annotated[int, operator.add]
    step: Annotated[List[str], operator.add]


# 节点A
def node_a(state: OverallState) -> OverallState:
    print(f"进入节点A")
    return {"value": 1, "step": ["A"]}


# 节点B
def node_b(state: OverallState) -> OverallState:
    print(f"进入节点B")
    return {"value": 1, "step": ["B"]}


# 节点C
def node_c(state: OverallState) -> OverallState:
    print(f"进入节点C")
    return {"value": 1, "step": ["C"]}


# 定义条件路由函数 奇数跳转到node_c 偶数跳转到node_b
def route_condition(state: OverallState) -> Literal["node_b", "node_c"]:
    if state.get("value") % 2 == 0:
        return "node_b"
    else:
        return "node_c"


# 构建图
graph_builder = StateGraph(OverallState)

graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)
graph_builder.add_node("node_c", node_c)

# 启动
graph_builder.add_edge(START, "node_a")


graph_builder.add_conditional_edges(
    "node_a",
    path=route_condition,
    path_map={"node_b": "node_b", "node_c": "node_c"},
)

# 终止
graph_builder.add_edge("node_b", END)
graph_builder.add_edge("node_c", END)

graph = graph_builder.compile()
graph.invoke({"value": 1, "step": ["Start Node"]})

# 生成调用图
generate_mermaid_images(graph)

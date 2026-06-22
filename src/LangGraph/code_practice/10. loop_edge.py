import operator
from typing import List, Literal

from langgraph.constants import START, END
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import StateGraph
from typing_extensions import TypedDict, Annotated

from utils import generate_mermaid_images


class OverallState(TypedDict):
    count: Annotated[int, operator.add]
    max_count: int
    result: Annotated[List[str], operator.add]


# 节点A
def node_a(state: OverallState) -> OverallState:
    print(f"进入节点A处理任务")
    return {"count": 1, "result": ["A执行1次"]}


# 节点B
def node_b(state: OverallState) -> OverallState:
    print(f"进入节点B处理任务")
    return {"count": 1, "result": ["B执行1次"]}


# 路由函数
def loop_route(state: OverallState) -> Literal["node_b", END]:
    if state.get("count") >= state.get("max_count"):
        print(f"满足循环终止条件")
        return END
    else:
        print(f"不满足循环终止条件,继续执行")
        return "node_b"


# 构建图
graph_builder = StateGraph(OverallState)
graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)

# 注意这里的循环构建图方式 不需要END节点 因为在route函数中已经指定
graph_builder.add_edge(START, "node_a")
graph_builder.add_conditional_edges(source="node_a", path=loop_route)
graph_builder.add_edge("node_b", "node_a")


# 执行
graph = graph_builder.compile()

generate_mermaid_images(graph)

try:
    result = graph.invoke(
        {"count": 0, "max_count": 10, "result": []}, config={"recursion_limit": 5}
    )
    print(result)
except GraphRecursionError as error:
    print(f"递归错误: {error}")

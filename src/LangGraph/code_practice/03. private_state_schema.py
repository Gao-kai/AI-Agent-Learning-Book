from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

"""
当某些情况下有些数据只需要在图中的部分节点之间共享，除此之外的节点都不可共享数据，那么可以采用：
1. 任意节点都可共享的数据放在全局OverallState中
2. 特殊节点之间共享的数据放在私有的State中

其本质是LangGraph在底层对状态做了隔离：
1. 编译时 通过类型系统静态检查来告诉LangGraph每个节点期望收到的状态和返回的状态
2. 运行时 LangGraph通过状态合并策略来决定是否要将节点返回的字典合并更新到全局状态中（非全局状态字段不会处理合并策略）
3. 节点之间私有数据传递则是通过边和边之间的隐式传递来实现的
"""


# 定义全局共享状态
class OverallState(TypedDict):
    name: str
    age: int
    version: str
    id: str


# 定义节点A的输出NodeAOutput
class NodeAOutput(TypedDict):
    score: int
    degree: str


# 定义节点A（A和B节点共享私有数据 A的输出作为B的输入）
def node_a(state: OverallState) -> NodeAOutput:
    print(f"执行节点 nodeA")
    print(f"获取输入共享状态： {state}")
    output = {"score": 100, "degree": "A"}
    print(f"返回值为：{output}")
    return output


# 定义节点B的输入NodeBInput
class NodeBInput(TypedDict):
    score: int
    degree: str


# 定义节点B（A和B节点共享私有数据 A的输出作为B的输入）
def node_b(state: NodeBInput) -> OverallState:
    print(f"执行节点 nodeB")
    print(f"获取输入状态： {state}")
    output = {"name": "lilei", "age": 18}
    print(f"返回值为全局状态：{output}")
    return output


# 定义节点C (C节点只能访问全局状态，无法访问A和B之间的私有状态)
def node_c(state: OverallState) -> OverallState:
    print(f"执行节点 nodeC")
    print(f"获取输入状态： {state}")
    output = {"id": "001"}
    print(f"返回值为：{output}")
    return output


# 构建图
graph_builder = StateGraph(OverallState)
graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)
graph_builder.add_node("node_c", node_c)

graph_builder.add_edge(START, "node_a")
graph_builder.add_edge("node_a", "node_b")
graph_builder.add_edge("node_b", "node_c")
graph_builder.add_edge("node_c", END)

graph = graph_builder.compile()

graph.invoke({"name": "Tom", "age": 20, "version": "1.2", "id": "100"})

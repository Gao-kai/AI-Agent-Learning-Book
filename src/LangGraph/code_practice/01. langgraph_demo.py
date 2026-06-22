from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import os


# 定义状态State
class OverallState(TypedDict):
    user_input: str


# 定义节点Node
def demo_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    print(f"用户输入为:", user_input)
    return {"user_input": "我还好呢！"}


# 注册图
graph_builder = StateGraph(OverallState)

# 添加节点
graph_builder.add_node("demo_node", demo_node)

# 添加边
graph_builder.add_edge(START, "demo_node")
graph_builder.add_edge("demo_node", END)

# 编译图
graph = graph_builder.compile()

# 查看调用图
# 获取当前文件的主文件名作为输出图片名
current_filename = os.path.splitext(os.path.basename(__file__))[0]
draw_mermaid_path = os.path.join(
    os.path.dirname(__file__), "../mermaid_images", f"{current_filename}.png"
)
graph.get_graph().draw_mermaid_png(output_file_path=draw_mermaid_path)

# 调用图
result = graph.invoke({"user_input": "你今天还好吗"})

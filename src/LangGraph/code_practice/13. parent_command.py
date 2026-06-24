"""
当我们希望从一个子图中的某个节点导航到父图中的另一个节点时，可以：
1. 使用Command来实现跨子图导航
2. 指定graph=Command.PARENT
3. 更新的状态如果是子图和主图共享的状态时，需要为主图状态中的键指定一个reducer
"""

from typing import TypedDict, Annotated
from LangGraph.code_practice.utils import generate_mermaid_images
from langgraph.graph.state import StateGraph
from langgraph.types import Command
from langgraph.graph import START, END
from langgraph.pregel import Pregel


# 定义父图状态
class ParentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    task_status: str
    subtask_result: str


# 定义子图状态
class ChildState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    task_status: str
    subtask_result: str
    child_data: str


# 定义主agent-启动导航去子图
def main_agent(state: ParentState):
    print("执行主任务")
    return Command(
        update={
            "messages": [("system", "启动子任务")],
            "task_status": "subtask Start",
        },
        goto="sub_graph_node",
    )


# 定义主agent-结束节点
def task_finisher(state: ParentState):
    print("结束主任务")
    return Command(
        update={"messages": [("system", "主任务完成")], "task_status": "done"}
    )


# 定义子任务节点
def data_processor(state: ChildState):
    print("执行子任务")
    return Command(
        update={
            "messages": [("subtask", "子任务处理完成")],
            "subtask_result": "100",
            "task_status": "subtask Done",
        },
        goto="task_finisher",
        graph=Command.PARENT,
    )


# 创建子图 - 返回类型是 Pregel（compile后的类型）
def create_sub_graph() -> Pregel:
    builder = StateGraph(ChildState)
    builder.add_node("data_processor", data_processor)
    builder.add_edge(START, "data_processor")
    builder.add_edge("data_processor", END)
    return builder.compile()


# 创建主图 - 返回类型是 Pregel（compile后的类型）
def create_main_graph() -> Pregel:
    builder = StateGraph(ParentState)

    builder.add_node("main_agent", main_agent)
    builder.add_node("task_finisher", task_finisher)
    builder.add_node("sub_graph_node", create_sub_graph())

    builder.add_edge(START, "main_agent")
    builder.add_edge("main_agent", "sub_graph_node")
    return builder.compile()


graph = create_main_graph()
result = graph.invoke({"messages": [], "task_status": "start", "subtask_result": ""})
print(result)
generate_mermaid_images(graph)
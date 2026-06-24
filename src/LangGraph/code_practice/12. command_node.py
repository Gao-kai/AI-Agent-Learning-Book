"""
借助于Command API，我们可以：
1. 不用手动构建图的每一条固定的边，可以基于goto来动态路由到任意一个节点
2. 在动态路由的过程中还是可以更新全局状态，这一点Send做不到，Send只能给对应的节点发送消息，但是不能修改全局状态

当你在需要更新状态的同时导航到下一个节点，此时需要使用Command，必须多智能体系统中切换的同时传递消息
当你值需要在节点之间有条件的路由而不需要更新状态时，此时需要使用Condition Edges
"""

import operator
from typing import TypedDict, Annotated, List, Tuple

from langgraph.types import Command
from langgraph.graph import END, START, StateGraph

from LangGraph.code_practice.utils import generate_mermaid_images


# 定义全局状态
class AgentState(TypedDict):
    curr_agent: str
    messages: Annotated[List[Tuple[str, str]], operator.add]
    done: bool


# 定义主代理
def decision_agent(state: AgentState) -> Command[AgentState]:
    print("主代理开始执行任务")

    # 获取消息列表中最近一条消息内容
    last_message = state.get("messages")[-1]
    role = last_message[0]

    if role == "user":
        # 跳转到数学节点的同时更新全局状态
        if "数学" in last_message[1]:
            return Command(
                update={
                    "messages": [("system", "导航到数学代理")],
                    "curr_agent": "math_agent",
                    "done": False,
                },
                goto="math_agent",
            )
        # 跳转到翻译节点的同时更新全局状态
        elif "翻译" in last_message[1]:
            return Command(
                update={
                    "messages": [("system", "导航到翻译代理")],
                    "curr_agent": "translation_agent",
                    "done": False,
                },
                goto="translation_agent",
            )

    # 跳转到END节点
    return Command(
        update={
            "messages": [("system", "任务完成")],
            "done": True,
        },
        goto=END,
    )


# 定义解决数学问题的Agent
def math_agent(state: AgentState) -> Command[AgentState]:
    print("执行数学计算，记录计算结果")
    return Command(
        update={
            "messages": [("assistant", f"数学计算结果为{100}")],
            "curr_agent": "decision_agent",
        },
        goto="decision_agent",
    )


# 定义解决翻译问题的Agent
def translation_agent(state: AgentState) -> Command[AgentState]:
    print("执行数学计算，记录计算结果")
    return Command(
        update={
            "messages": [("assistant", "数学计算结果为100")],
            "curr_agent": "decision_agent",
        },
        goto="decision_agent",
    )


# 构建图
builder = StateGraph(AgentState)
builder.add_node(math_agent)
builder.add_node(decision_agent)
builder.add_node(translation_agent)

builder.add_edge(START, "decision_agent")
graph = builder.compile()
generate_mermaid_images(graph)
result = graph.invoke({"messages": [("user", "数学计算")]})
print(result)

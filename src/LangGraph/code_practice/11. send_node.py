"""
Send 用于将消息直接发送给指定的节点，但是不更新全局状态
1. Send可以实现一对多的消息转发，从而实现给多个节点发送消息而不更新全局状态
[Send("A",data1),Send("B",data2)]

2. Send可以实现动态路由，不更新全局状态

一般情况下从当前节点跳转到下一个节点的时候：
1. 上一个节点return的数据会去更新全局状态State
2. 更新后的State会当作参数传入下一个节点
3. 如果return的字典中包含了原本不在全局State中的字段，那么会被截取掉

但是通过使用Send API的方法，我们可以：
1. A节点可以在路由到B节点的时候通过return一个Send方法来实现路由跳转
2. Send方法不仅可以包含目标节点，而且可以携带AB两个节点之间通信的私有数据
3. A节点在通过Send节点跳转的时候不会更新全局状态
"""

import operator

from typing import TypedDict, Annotated, List, Sequence

from langgraph.graph.state import StateGraph, START, END
from langgraph.types import Send

from LangGraph.code_practice.utils import generate_mermaid_images


# 定义全局状态
class OverallState(TypedDict):
    subjects: List[str]
    jokes: Annotated[List[str], operator.add]


# 定义生成类型节点
def generate_types(state: OverallState):
    return {"subjects": ["Apple", "Banana", "StrewBerry"]}


# 定义为每个主题生成输出节点
def generate_jokes(state):
    subject = state.get("subject", "--")
    jokes_map = {
        "Apple": "这是Apple的笑话",
        "Banana": "这是Banana的笑话",
        "StrewBerry": "这是StrewBerry的笑话",
    }
    joke = jokes_map.get(subject)
    return {"jokes": [joke]}


# 定义条件边函数
def map_subjects_to_jokes(state: OverallState) -> Sequence[Send]:
    subjects = state.get("subjects")
    send_list = []
    for subject in subjects:
        send_list.append(Send("generate_jokes", {"subject": subject}))
    return send_list


# 演示Send节点使用方法
def main():
    graph_builder = StateGraph(OverallState)

    graph_builder.add_node("generate_types", generate_types)
    graph_builder.add_node("generate_jokes", generate_jokes)

    graph_builder.add_edge(START, "generate_types")
    graph_builder.add_conditional_edges("generate_types", map_subjects_to_jokes)
    graph_builder.add_edge("generate_jokes", END)

    graph = graph_builder.compile()
    generate_mermaid_images(graph)
    result = graph.invoke({"subjects": [], "jokes": []})
    print(f"最终执行结果为: {result}")


if __name__ == "__main__":
    main()

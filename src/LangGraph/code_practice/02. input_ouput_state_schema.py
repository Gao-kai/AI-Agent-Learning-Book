"""
LangGraph 图输入输出模式和私有状态传递演示
该演示展示了：
1. 如何定义图的输入和输出模式
2. 如何在节点间传递私有状态
"""

"""
LangGraph 图输入输出模式
1. 如何分别定义图的输入和输出Schema
2. 输入的Schema限制了只有符合schema中的字段才可以图解析后处理
3. 输出的Schema限制了只有符合schema中的字段才可以被图输出后返回
4. 如果不指定默认就等于State Schema
"""

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# 定义输入状态模式
class InputState(TypedDict):
    question: str


# 定义输出状态模式
class OutputState(TypedDict):
    answer: str


# 定义整体状态模式，结合输入和输出
class OverallState(InputState, OutputState):
    pass


""" 
处理输入并生成答案的节点

Args:
    state: 输入状态
Returns:
    dict: 包含答案的字典
"""


def answer_node(state: InputState):

    print(f"执行 answer_node 节点:")
    # 输入: {'question': '你好'} 只接受InputState中字段 多余字段不会被解析为state
    print(f" 输入: {state}")
    answer = (
        "再见下次再聊"
        if "bye" in state["question"].lower()
        else "你好，有什么可以帮助的"
    )
    result = {"answer": answer, "done": True}
    print(f" 输出: {result}")
    return result


"""
输入输出模式
"""


def demo_input_output_schema():
    print("=== 演示输入输出模式 ===")

    # 使用指定的输入和输出模式构建图 可以保证入参和出参的结构一定符合Schema的要求
    builder = StateGraph[OverallState, None, InputState, OutputState](
        OverallState, input_schema=InputState, output_schema=OutputState
    )
    # 添加答案节点
    builder.add_node("answer_node", answer_node)
    builder.add_edge(START, "answer_node")
    builder.add_edge("answer_node", END)

    # 编译图
    graph = builder.compile()

    # 使用输入调用图并打印结果
    result = graph.invoke({"question": "你好", "env": "开发环境", "version": "1.0.0"})

    # 图调用结果: {'answer': '你好，有什么可以帮助的'} 只接受OutputSchema的字段
    print(f"图调用结果: {result}")
    print()


# 主函数
def main():
    print("=== LangGraph 图输入输出模式限制参数 ===\\n")
    demo_input_output_schema()
    print("=== 演示完成 ===")


if __name__ == "__main__":
    main()

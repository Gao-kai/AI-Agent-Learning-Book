"""
本案例演示了使用Command的几个优点：
1. 状态更新和流程跳转一体化，在外部第三方工具执行后基于工具执行结果决定去哪一个节点
2. 避免添加多余的条件边，只需要在goto中指定下一步去哪
3. 清晰的控制流和灵活的状态管理
"""

import time
from typing import TypedDict, Dict, Annotated, List
import operator

from langgraph.types import Command
from langgraph.graph import START, END, StateGraph


# 第一步：定义全局状态
class OverallState(TypedDict):
    id: str
    user_info: Dict
    messages: Annotated[List[str], operator.add]
    resolved: bool


# 模拟数据库
CUSTOMER_DATABASE = {
    "CUST001": {
        "name": "张三",
        "email": "zhangsan@example.com",
        "membership_level": "金牌会员",
        "account_status": "正常",
    },
    "CUST002": {
        "name": "李四",
        "email": "lisi@example.com",
        "membership_level": "银牌会员",
        "account_status": "正常",
    },
    "CUST003": {
        "name": "王五",
        "email": "wangwu@example.com",
        "membership_level": "普通会员",
        "account_status": "欠费",
    },
}


# 定义工具函数：查询客户信息
def search_user_info(id: str) -> Dict:
    print("模拟查询用户数据库")
    time.sleep(3)
    user_info = CUSTOMER_DATABASE.get(id, {})
    if user_info:
        print(f"查询到客户信息为{user_info}")
        return user_info
    else:
        print("查询失败")
        return {"error": "用户不存在"}


# 定义节点：查询DB Agent
def user_search_agent(state: OverallState) -> Command[OverallState]:
    user_id = state.get("id")
    user_info = search_user_info(user_id)
    return Command(
        update={"user_info": user_info, "messages": [f"查询到客户信息为{user_info}"]},
        goto="support_agent",
    )


# 定义节点：客服处理 Agent
def support_agent(state: OverallState) -> Command[OverallState]:
    user_info = state.get("user_info")
    messages = state.get("messages")

    if "error" in user_info:
        response = "抱歉无法查询到你的客户信息"
        next_node = END
    else:
        # 根据客户等级提供个性化服务
        membership_level = user_info.get("membership_level", "未知")
        name = user_info.get("name", "客户")

        if membership_level == "金牌会员":
            response = (
                f"尊敬的金牌会员{name}，您好！我们非常重视您的问题，将优先为您处理。"
            )
        elif membership_level == "银牌会员":
            response = f"{name}您好！我们会尽快为您解决问题。"
        else:
            response = f"{name}您好！感谢您的咨询。"

        # 模拟处理问题
        response += "\n\n 我们已经收到您的问题，正在为您处理..."
        next_node = "issue_resolver"

    return Command(update={"messages": [response]}, goto=next_node)


# 定义节点：问题处理 Agent
def issue_resolver(state: OverallState) -> Command[OverallState]:
    print("正在解决问题")
    time.sleep(1)
    return Command(update={"messages": ["问题解决了"], "resolved": True}, goto=END)


# 构建图
builder = StateGraph(OverallState)
builder.add_node(user_search_agent)
builder.add_node(support_agent)
builder.add_node(issue_resolver)

builder.add_edge(START, "user_search_agent")
graph = builder.compile()
result = graph.invoke(
    {"id": "CUST001", "user_info": {}, "messages": [], "resolved": False}
)
print(result)

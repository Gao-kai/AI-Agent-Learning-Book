"""
LangGraph中的节点重试

场景：当节点执行的是网络请求等操作时可能会因为网络连接不稳定导致API请求失败，此时可以给节点设置重试机制
默认：默认情况下除了遇到以下异常之外都会重试（也就是已经确定充实也没有用的错误）
 ValueError
 TypeError
 ArithmeticError
 ImportError
 LookupError
 NameError
 SyntaxError
 RuntimeError
 ReferenceError
 StopIteration
 StopAsyncIteration

自定义重试策略：只对节点raise出来的异常对象中的某些情况进行重试，对于不满足条件的直接报错
比如：对于code为9开头的报错重试，对于其他数字开头的报错不重试
"""

import time
from typing import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import START, END, StateGraph
from langgraph.types import CachePolicy, RetryPolicy


class OverallState(TypedDict):
    message: str
    code: str


# 全局变量 计数器
request_count = 0


# API服务请求节点
def http_request_node(state: OverallState):
    global request_count
    request_count += 1

    print(f"第{request_count}次调用API")

    if request_count <= 3:
        raise Exception({"message": "API请求失败", "code": "9999"})
    else:
        return {"message": "API请求成功", "code": "0000"}


# 自定义重试策略
# LangGraph 会在节点抛出异常时调用你的自定义重试策略函数，并 自动传入异常对象 作为参数。
def custom_retry_policy(exception: Exception) -> bool:
    error_data = exception.args[0]
    code = error_data.get("code", "")

    # 如果错误码以9开头 则返回True 重试一次；否则不重试
    if code.startswith("9"):
        return True
    else:
        return False


# 构建图
graph_builder = StateGraph(OverallState)


# 添加自定义节点重试策略
graph_builder.add_node(
    "http_request_node",
    http_request_node,
    retry_policy=RetryPolicy(max_attempts=10, retry_on=custom_retry_policy),
)
graph_builder.add_edge(START, "http_request_node")
graph_builder.add_edge("http_request_node", END)

# 编译图
graph = graph_builder.compile()
print(f"返回结果为：{graph.invoke({})}")

"""
第1次调用API
第2次调用API
第3次调用API
第4次调用API
返回结果为：{'message': 'API请求成功', 'code': '0000'}
"""

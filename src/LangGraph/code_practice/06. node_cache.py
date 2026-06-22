"""
LangGraph中的节点Node支持设置Node节点级别的缓存
当某个Node节点（本质是一个函数）的任务是昂贵的计算，并且这个计算结果在输入的参数固定的情况返回值不变
此时就可以给这个Node节点设置缓存策略

LangGraph 的缓存机制需要两个部分：

1. 缓存策略 （ cache_policy ）：定义缓存的有效期 在添加节点的时候指定
2. 缓存存储 （如 MemorySaver ）：实际存储缓存数据的地方 在编译图的时候指定
"""

import time
from typing import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import START, END, StateGraph
from langgraph.types import CachePolicy


class OverallState(TypedDict):
    value: int
    total_result: int


# 昂贵计算节点
def expensive_calc_node(state: OverallState):
    time.sleep(2)
    print("等待复杂计算中⌛️....")
    value = state.get("value")
    return {"total_result": value * 100}


# 构建图
graph_builder = StateGraph(OverallState)

# 在添加节点的时候声明cache_policy 参数为ttl 不设置代表永不过期
graph_builder.add_node(
    "expensive_calc_node", expensive_calc_node, cache_policy=CachePolicy(ttl=60 * 60)
)
graph_builder.add_edge(START, "expensive_calc_node")
graph_builder.add_edge("expensive_calc_node", END)

# 编译图
graph = graph_builder.compile(cache=InMemoryCache())

print(f"首次执行计算的结果为：{graph.invoke({"value": 100})}")
print(f"第二次执行计算的结果为：{graph.invoke({"value": 100})}")
print(f"第三次执行计算的结果为：{graph.invoke({"value": 200})}")

"""
输出依次为：

等待复杂计算中⌛️....
首次执行计算的结果为：{'value': 100, 'total_result': 10000}

第二次执行计算的结果为：{'value': 100, 'total_result': 10000} => 说明中间这一次计算在参数相同的时候缓存生效

等待复杂计算中⌛️....
第三次执行计算的结果为：{'value': 200, 'total_result': 20000}
"""

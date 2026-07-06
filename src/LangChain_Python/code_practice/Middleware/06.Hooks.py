"""
=============================
LangChain Agent Middleware Hooks 完整指南
=============================

Agent定义的多个中间件
before是按照传入的顺序依次执行
after是按照传入的顺序倒序执行
wrap是洋葱圈模型

一、Hook 概览

| Hook | 执行时机 | 作用 | 返回值 |
|------|----------|------|--------|
| before_agent | Agent 执行开始前 | 初始化、检查前置条件 | dict / Command / None |
| after_agent | Agent 执行完成后 | 清理、最终处理、日志 | dict / Command / None |
| before_model | 模型调用前 | 修改请求、条件跳转 | dict / Command / None |
| after_model | 模型调用后 | 处理响应、条件跳转 | dict / Command / None |
| dynamic_prompt | 模型调用前（动态生成） | 动态生成系统提示词 | str / SystemMessage |
| wrap_model_call | 模型调用期间 | 完全控制模型执行流程 | ModelResponse / AIMessage |
| wrap_tool_call | 工具调用期间 | 完全控制工具执行流程 | ToolMessage / Command |


二、生命周期执行顺序

Agent.invoke()
    │
    ▼
┌─────────────────────────────────────────────┐
│  before_agent ──── 初始化、前置检查         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  主循环开始                                  │
│  ┌───────────────────────────────────────┐  │
│  │  before_model ─── 修改请求、条件跳转    │  │
│  └───────────────────────────────────────┘  │
│         │                                   │
│         ▼                                   │
│  ┌───────────────────────────────────────┐  │
│  │  wrap_model_call ─── 模型调用拦截      │  │
│  │  (可多次调用、跳过、重试)              │  │
│  └───────────────────────────────────────┘  │
│         │                                   │
│         ▼                                   │
│  ┌───────────────────────────────────────┐  │
│  │  after_model ──── 处理响应、条件跳转   │  │
│  └───────────────────────────────────────┘  │
│         │                                   │
│         ▼ (如需工具调用)                     │
│  ┌───────────────────────────────────────┐  │
│  │  wrap_tool_call ─── 工具调用拦截       │  │
│  │  (可多次调用、跳过、重试)              │  │
│  └───────────────────────────────────────┘  │
│         │                                   │
│         ▼ (循环直到结束)                     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  after_agent ───── 清理、最终处理          │
└─────────────────────────────────────────────┘

三、各 Hook 详细说明

1. before_agent
----------------
执行时机：Agent 开始执行之前（整个流程只执行一次）
适用场景：
    - 初始化全局状态
    - 检查用户权限
    - 记录会话开始时间

示例：
    @before_agent
    def init_session(state: AgentState, runtime: Runtime) -> dict:
        return {"session_start_time": datetime.now().isoformat()}

2. after_agent
--------------
执行时机：Agent 执行完成之后（整个流程只执行一次）
适用场景：
    - 清理资源
    - 记录会话结束时间
    - 生成最终报告

示例：
    @after_agent
    def cleanup_session(state: AgentState, runtime: Runtime) -> None:
        print(f"会话结束，共处理 {len(state['messages'])} 条消息")

3. before_model
---------------
执行时机：每次模型调用之前
适用场景：
    - 修改请求参数
    - 条件跳转（提前结束、跳过模型）
    - 记录模型调用前状态

示例：
    @before_model(can_jump_to=["end"])
    def check_length(state: AgentState, runtime: Runtime) -> dict | None:
        if len(state["messages"]) > 100:
            return {"jump_to": "end"}  # 提前结束
        return None

4. after_model
--------------
执行时机：每次模型调用之后
适用场景：
    - 处理模型响应
    - 根据响应内容条件跳转
    - 记录模型响应

示例：
    @after_model(can_jump_to=["tools", "end"])
    def handle_response(state: AgentState, runtime: Runtime) -> dict | None:
        last_msg = state["messages"][-1]
        if "总结" in last_msg.content:
            return {"jump_to": "end"}
        return None

5. dynamic_prompt
-----------------
执行时机：模型调用前（基于 wrap_model_call）
适用场景：
    - 根据上下文动态生成系统提示词
    - 个性化提示词

示例：
    @dynamic_prompt
    def personalized_prompt(request: ModelRequest) -> str:
        user_id = request.runtime.context.get("user_id")
        user_info = get_user_info(user_id)
        return f"你是 {user_info.name} 的专属助手，你的性格是 {user_info.personality}"

6. wrap_model_call
------------------
执行时机：模型调用期间（最强大的 hook，AOP编程核心）
适用场景：
    - 重试机制 超出设置最大重试次数直接终止
    - 缓存（短路跳过）获取模型调用参数，如果hash一致则直接从缓存获取
    - 修改请求/响应
    - 错误处理
参数：
    - request 请求对象 包含了模型请求的所有信息
    - handler 处理器，用于处理请求返回调用结果

示例：
    from langchain.agents.middleware import wrap_model_call

    @wrap_model_call
    def retry_on_error(request: ModelRequest, handler) -> ModelResponse:
        for attempt in range(3):
            try:
                return handler(request)
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"重试第 {attempt + 1} 次")

7. wrap_tool_call
-----------------
执行时机：工具调用期间
适用场景：
    - 工具调用重试
    - 工具调用日志
    - 修改工具参数
    - 错误处理

示例：
    from langchain.agents.middleware import wrap_tool_call

    @wrap_tool_call
    def log_tool_call(request: ToolCallRequest, handler) -> ToolMessage:
        print(f"调用工具: {request.tool_call['name']}")
        result = handler(request)
        print(f"工具结果: {result.content}")
        return result

8. hook_config
--------------
辅助装饰器：配置 hook 的行为
适用场景：
    - 指定 hook 可以跳转的目标节点

示例：
    @hook_config(can_jump_to=["end", "tools"])
    @before_model
    def conditional_hook(state: AgentState, runtime: Runtime) -> dict | None:
        if should_skip(state):
            return {"jump_to": "end"}
        return None

四、关键区别总结

| 维度 | before/after_* | wrap_* | dynamic_prompt |
|------|------------------|----------|------------------|
| 控制权 | 有限（只能修改状态） | 完全控制（可跳过/重试） | 中等（仅修改提示词） |
| 执行频率 | 每次/每个流程一次 | 每次调用期间 | 每次模型调用前 |
| 参数 | state, runtime | request, handler | request |
| 返回值 | dict/Command/None | ModelResponse/ToolMessage | str/SystemMessage |
| 典型用途 | 日志、状态管理 | 重试、缓存、错误处理 | 动态提示词 |

五、使用建议

1. 日志记录 → 使用 before_model / after_model
2. 重试机制 → 使用 wrap_model_call / wrap_tool_call
3. 条件跳过 → 使用 before_model + can_jump_to
4. 动态提示词 → 使用 dynamic_prompt
5. 会话管理 → 使用 before_agent / after_agent

这些 hooks 可以组合使用，形成强大的中间件链，实现复杂的 Agent 行为控制！
"""

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentState, AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from LangChain_Python.models.chat_model import model

"""
基于类实现中间件和基于装饰器实现
本质都是实现了一个AgentMiddleware的子类，并重写该类的钩子函数
因此除了单个hook快速挂载到agent生命周期的某个节点上，其他的情况一律推荐用类的写法实现
1. 本身装饰器的底层就是类AgentMiddleware的实现
2. 类可以将多个hook归纳到一个类里面，便于维护

以装饰器为例：

1. 装饰器所修饰的函数参数：
    参数state Langgraph中的全局状态
    参数runtime 维护Agent运行过程中的上下文环境

2. 装饰器所修饰的函数返回值：
    None 不修改Agent的状态
    字典：更新Agent中state的状态，返回的字典中支持：
        - jump_to键控制流程跳转 {"jump_to": "end"}
        - messages 用于添加、替换或修改消息列表
        - 其他自定义key来更新图的状态

3. 要想jump_to生效，必须传入装饰器参数can_jump_to
    can_jump_to: 指定 hook 可以跳转的目标节点
    默认值为 ["end"] ：跳转到 Agent 执行结束（或第一个 after_agent 钩子）
    可以选择跳转到工具tools：跳转到工具节点
    或者选择跳转到model节点：跳转到模型节点（或第一个 before_model 钩子）
"""


class LogMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime):
        print("before_agent触发调用")
        return {"jump_to": "end", "messages": []}

    def before_model(self, state: AgentState, runtime: Runtime):
        print("before_model触发调用")
        return None

    def after_model(self, state: AgentState, runtime: Runtime):
        print("after_model触发调用")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime):
        print("after_agent触发调用")
        return None


# 创建agent
agent = create_agent(
    model=model,
    tools=[],
    middleware=[LogMiddleware()],
    system_prompt="你是一个AI代码助手",
)

# 调用agent
response = agent.invoke({"messages": [HumanMessage("你好")]})

# 查看响应
for msg in response["messages"]:
    msg.pretty_print()

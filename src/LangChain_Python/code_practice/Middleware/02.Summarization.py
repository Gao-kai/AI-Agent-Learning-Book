"""
中间件Summarization
对历史消息进行摘要，当达到trigger的条件的时候，调用大模型对历史消息进行总结，
并将摘要的消息包装为一个HumanMessage，放在消息列表的最顶部，再加上keep保留的消息列表
一起发送给大模型，等待大模型返回响应。


"""

from langchain.agents.factory import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from src.LangChain_Python.models.chat_model import model

# 创建Agent
agent = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            # 摘要模型，在传入fraction时必须配置max_input_tokens
            model=model,
            # 触发条件 可以多选 任意一个条件满足就开始总结
            trigger=[("tokens", 3000), ("messages", 6), ("fraction", 0.001)],
            # 保留的历史消息
            keep=("messages", 2),
            # token计算函数
            token_counter=count_tokens_approximately,
            # 给大模型看的提示词，大模型最终输出不一定看这个 主要是把英文转化为中文
            summary_prompt="上下文会话即将超出模型上下文长度，开始总结\n{messages}",
            # 生成的摘要的最大token字符
            trim_tokens_to_summarize="3000",
        )
    ],
    system_prompt="你是一个智能AI助手，可以解决很多问题，自己不知道的可以直接说不知道，不要乱回答",
)

# 模拟消息列表
messages = [
    HumanMessage("你好"),
    AIMessage("你好，我有什么可以帮你"),
    HumanMessage("你帮我计算下1+100等于多少"),
    AIMessage("看起来是101"),
    HumanMessage("北京到上海坐火车怎么走？"),
    AIMessage("稍等我查询下，一般是坐高铁"),
]

# 调用Agent
response = agent.invoke({"messages": messages})

# 查看响应
for msg in response["messages"]:
    msg.pretty_print()

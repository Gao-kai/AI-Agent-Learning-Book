"""
中间件HumanInLoop

在工具调用前可以配置是否中断执行，等待用户决策后继续执行，或者结束工具调用。
可用的决策有：
1. approve：用户同意当前操作
2. reject：用户拒绝当前操作
3. edit：用户编辑当前操作

"""

from jinja2.utils import consume
from langchain.agents.factory import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from openai.types.responses.response_computer_tool_call_param import ActionClick
from src.LangChain_Python.models.chat_model import model
from langchain.tools import tool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    city: str = Field(description="城市名称")


# 查询天气工具
@tool(args_schema=WeatherInput, description="查询指定城市的天气")
def get_weather(city: str):
    return f"当前{city}天气晴朗，气温28摄氏度，空气质量优。"


class NewsInput(BaseModel):
    category: str = Field(description="新闻类别，如科技、体育、娱乐")


# 查询新闻工具
@tool(args_schema=NewsInput, description="查询指定类别的新闻")
def get_news(category: str):
    return f"【{category}新闻】今日热点：人工智能技术取得重大突破，市场反应积极。"


class SendEmailInput(BaseModel):
    to: str = Field(description="收件人邮箱")
    subject: str = Field(description="邮件主题")
    content: str = Field(description="邮件内容")


# 发送邮件工具
@tool(args_schema=SendEmailInput, description="发送邮件给指定收件人")
def send_email(to: str, subject: str, content: str):
    return f"邮件已成功发送至 {to}，主题：{subject}，内容已送达。"


class ReadEmailInput(BaseModel):
    folder: str = Field(description="邮箱文件夹，如收件箱、已发送")


# 读取邮件工具
@tool(args_schema=ReadEmailInput, description="读取指定文件夹的邮件")
def read_email(folder: str):
    return f"【{folder}邮件列表】\n1. 发件人：admin@example.com，主题：会议通知\n2. 发件人：support@company.com，主题：技术支持请求"


# 创建agent
agent = create_agent(
    model=model,
    tools=[get_weather, get_news, send_email, read_email],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_weather": True,
                "get_news": False,
                "send_email": {"allowed_decisions": ["approve", "edit"]},
                "read_email": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "读取文件工具中断，等待用户操作",
                },
            },
            description_prefix="工具调用中断，请操作...",
        )
    ],
)

# 创建config
config = {"configurable": {"thread_id": "0001"}}

# 调用Agent
response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                "帮我查询徐州的天气，读取app目录下邮件注意这个和后面的发送没有任何关系就是读取一下就可以，然后发送给lima用户，最后再帮我查询科技新闻，四个事情可以一起做"
            )
        ]
    },
    config=config,
)


# 查看响应
for msg in response["messages"]:
    msg.pretty_print()


# 模拟用户决策为编辑
get_weather_decision = {"type": "edit", "edited_action": {"args": {"city": "上海"}}}

# 模拟用户同意操作
read_email_decision = {
    "type": "approve",
}

# 模拟用户同意操作
send_email_decision = {
    "type": "approve",
}

decision = {"decisions": []}


if response.get("__interrupt__"):
    action_requests = response["__interrupt__"][0].value["action_requests"]
    for action in action_requests:
        if action["name"] == "get_weather":
            decision["decisions"].append(get_weather_decision)
        elif action["name"] == "read_email":
            decision["decisions"].append(read_email_decision)
        elif action["name"] == "send_email":
            decision["decisions"].append(send_email_decision)

    # 决策后继续执行 注意这里resume接受的是一个字典对象，里面包含“decision”字段即可
    res = agent.invoke(
        Command(resume=decision),
        config=config,
    )

    print(res)

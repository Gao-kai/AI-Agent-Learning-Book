"""
PII 中间件
在调用模型时用于检测和处理对话中的个人身份信息，支持自定义处理策略

1. pii_type：指定要检测的个人身份信息类型，可选值包括：
    - "email"：检测邮箱地址
    - "credit_card"：检测信用卡号
    - "url"：检测URL
    - "mac_address"：检测MAC地址
    - "ip"：检测IP地址

2. strategy：指定处理策略，可选值包括：
    - "redact"：用固定字符串替换
    - "mask"：用星号替换个人身份信息
    - "hash"：用哈希值替换个人身份信息
    - "block"：直接报错

3. apply_to_input：在模型调用开始前检测
4. apply_to_output：在模型调用结束后检测
5. apply_to_tool_result：在工具调用结束后检测

你好，
我的邮箱是 [REDACTED_EMAIL]
我的信用卡号是 ************4242
我的IP地址是 <ip_hash:2a39f1ee>
我的Mac地址时 <mac_address_hash:f57b6b8d>
我的URL是 [MASKED_URL]
"""

from langchain.agents import create_agent
from langchain.agents.middleware.pii import PIIMiddleware
from langchain_core.messages.human import HumanMessage

from LangChain_Python.models.chat_model import model

# 创建agent
agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        PIIMiddleware(pii_type="email", strategy="redact", apply_to_input=True),
        PIIMiddleware(pii_type="credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware(pii_type="ip", strategy="hash", apply_to_input=True),
        PIIMiddleware(pii_type="mac_address", strategy="hash", apply_to_input=True),
        PIIMiddleware(pii_type="url", strategy="mask", apply_to_input=True),
    ],
)


# 调用agent
response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                "你好，我的邮箱是 admin@example.com，我的信用卡号是 4242424242424242，我的IP地址是 192.168.1.100，我的Mac地址时 00:1A:2B:3C:4D:5E，我的URL是 https://www.example.com"
            )
        ]
    }
)

# 查看响应
for msg in response["messages"]:
    msg.pretty_print()

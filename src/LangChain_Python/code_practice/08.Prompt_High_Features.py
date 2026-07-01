"""
本章节主要展示ChatPromptTemplate的高级特性

1. 基于partial实现变量预填充
2. 消息占位符
    - 在多轮历史对话系统中存储历史消息
    - 在Agent中间步骤处理
    - 当你希望在调用大模型的时候插入一些之前的历史消息列表
3. 消息占位符的通过placeholder关键字实现
4. 消息占位符通过MessagesPlaceholder指定变量名实现

"""

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder

## 1. 基于partial实现变量预填充
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个{role}助手，主要的用户是{person}"),
        ("user", "你好，我的问题是：{user_input}"),
    ]
)

chat_prompt_final = chat_prompt_template.partial(role="旅游", person="游客")

result1 = chat_prompt_final.invoke("北京故宫怎么玩？")
result2 = chat_prompt_final.invoke("北京长城怎么玩？")
print(result1)
print(result2)


## 2. 基于placeholder实现历史消息占位符

template = ChatPromptTemplate.from_messages(
    [("system", "你是一个旅游助手"), ("placeholder", "{history_message}")]
)

# 可以将历史会话消息直接填充到模版中供大模型调用
result3 = template.invoke(
    {
        "history_message": [
            HumanMessage("北京故宫怎么玩呢？"),
            AIMessage("稍等我查询一下给你答复"),
            HumanMessage("顺带帮我查下从上海去北京的路线"),
        ]
    }
)

print(result3)


## 3. 基于MessagesPlaceholder实现历史消息占位符

template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个旅游助手"),
        MessagesPlaceholder(variable_name="messages_list"),
        ("ai", "{question}"),
    ]
)

# 可以将历史会话消息直接填充到模版中供大模型调用
result4 = template.invoke(
    {
        "messages_list": [
            HumanMessage("北京故宫怎么玩呢？"),
            AIMessage("稍等我查询一下给你答复"),
            HumanMessage("顺带帮我查下从上海去北京的路线"),
        ],
        "question": "顺带再帮我查下北京故宫的门票",
    }
)

print(result4)

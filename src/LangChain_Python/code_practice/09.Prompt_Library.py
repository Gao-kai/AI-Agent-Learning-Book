"""
模版库

1. 可以提前基于产品经理提供的提示词模版，预先准备好模版文件，放在templates目录下统一维护
2. 调用时只需要传入变量即可
3. 可以将不同角色和业务场景的提示词模版提前写好 然后在使用的时候进行调用 不用每次都写很长的模版
"""

import os
import dotenv

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv(override=True)


# 新建模版提示词类（也可以通过单独文件目录进行管理）
class TemplateLibrary:

    # 代码审查专家模版
    CODE_REVIEWER = ChatPromptTemplate.from_messages(
        [
            ("system", "你是{language}代码专家，重点关注{focus}"),
            ("user", "审查代码：\n```{language}\n{code}\n```"),
        ]
    )

    # 翻译模版
    TRANSLATOR = ChatPromptTemplate.from_messages(
        [
            ("system", "你是翻译专家，精通{source}语言和{target}"),
            ("user", "翻译文本：{text}"),
        ]
    )


# 生成翻译模版提示词
translator_template = TemplateLibrary.TRANSLATOR.partial(
    source="法语", target="汉语"
).invoke(
    {
        "text": "你今天吃饭了吗？",
    }
)
print(translator_template)
# 模型调用

MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化Model
model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

result = model.invoke(translator_template)
print(result)

/**
 * 本章节主要讨论内存管理中总结策略（summarization）
 * 当上下文历史对话记录超出最大token时
 * 只保留最近的n个token的消息记录
 * 其余的消息需要总结后放入消息
 */
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import {
  AIMessage,
  BaseMessage,
  getBufferString,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { getEncoding } from "js-tiktoken";

const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0.2,
  maxTokens: undefined,
  timeout: undefined,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

async function summarizeHistoryMessage() {
  const MAX_TOKEN = 200;
  const MAX_STORED_TOKEN = 80;
  const history = new InMemoryChatMessageHistory();

  /**
   * 1. 创建历史消息记录并全部添加至Memory内存中
   */
  const messages = [
    { type: "human", content: "我想学做红烧肉，你能教我吗？" },
    {
      type: "ai",
      content:
        "当然可以！红烧肉是一道经典的中式菜肴。首先需要准备五花肉、冰糖、生抽、老抽、料酒等材料。",
    },
    { type: "human", content: "五花肉需要切多大块？" },
    {
      type: "ai",
      content:
        "建议切成3-4厘米见方的块，这样既容易入味，口感也更好。切好后可以用开水焯一下去除血沫。",
    },
    { type: "human", content: "炒糖色的时候有什么技巧吗？" },
    {
      type: "ai",
      content:
        "炒糖色是关键步骤。用小火慢慢炒，等冰糖完全融化变成焦糖色，冒小泡时就可以下肉了。注意不要炒过头，否则会发苦。",
    },
    { type: "human", content: "需要炖多长时间？" },
    {
      type: "ai",
      content:
        "一般需要炖40-60分钟，用小火慢炖，直到肉变得软糯入味。可以用筷子戳一下，能轻松戳透就说明好了。",
    },
    { type: "human", content: "最后收汁的时候要注意什么？" },
    {
      type: "ai",
      content:
        "收汁时要用大火，不断翻动，让汤汁均匀包裹在肉块上。看到汤汁变得浓稠，颜色红亮就可以出锅了。",
    },
  ];

  for (const message of messages) {
    if (message.type === "ai") {
      await history.addMessage(new AIMessage(message.content));
    } else if (message.type === "human") {
      await history.addMessage(new HumanMessage(message.content));
    }
  }

  /**
   * 2. 计算当前所有历史消息的token总和是否超出最大上下文token总和
   * 基于编码名称来获取编码对象
   * 编码对象包含encode和decode方法来计算token
   */
  const totalMessages = await history.getMessages();
  const totalTokens = countTokens(totalMessages);

  /**
   * 3. 判断当前totalTokens是否大于设定的阈值maxToken 此时需要对上下文消息进行总结
   * - 哪些消息需要进行总结
   * - 哪些消息需要进行保留
   *
   */
  if (totalTokens > MAX_TOKEN) {
    let toStoredMessages = [];
    let storedTokens = 0;

    for (let i = totalMessages.length - 1; i >= 0; i--) {
      const message = totalMessages[i];
      const messageToken = countToken(message);
      if (storedTokens + messageToken <= MAX_STORED_TOKEN) {
        // 头部入栈 保证消息顺序一致
        toStoredMessages.unshift(message);
        storedTokens += messageToken;
      } else {
        break;
      }
    }

    let toSummarizeMessages = totalMessages.slice(
      0,
      totalMessages.length - toStoredMessages.length,
    );
    let toSummarizeTokens = countTokens(toSummarizeMessages);

    console.log(`⚠️ 上下文历史消息Token总量超过阈值，开始总结...`);
    console.log(
      `✅ 需要保存的历史消息为${toStoredMessages.length}条，累计token为${storedTokens}个`,
    );
    console.log(
      `✅ 需要总结的历史消息为${toSummarizeMessages.length}条，累计token为${toSummarizeTokens}个`,
    );

    /**
     * 4. 将即将丢弃的历史消息进行总结
     */
    const summaryMessage = await summaryHistoryMessage(toSummarizeMessages);
    console.log("✅历史消息总结为:\n", summaryMessage);

    /**
     * 5. 清空历史记录
     * - 添加系统消息
     * - toStoredMessages 栈结构 先进栈的后出来 保证了总结前后被保存的最近的几条历史消息顺序一致
     */
    await history.clear();
    await history.addMessage(new SystemMessage(summaryMessage));
    for (const message of toStoredMessages) {
      await history.addMessage(message);
    }
  }
}

function countTokens(messages: BaseMessage[]) {
  const encoder = getEncoding("cl100k_base");

  let totalTokens = 0;

  for (const message of messages) {
    const content =
      typeof message.content === "string"
        ? message.content
        : JSON.stringify(message.content);
    const token = encoder.encode(content);
    totalTokens += token.length;
  }

  return totalTokens;
}

function countToken(message: BaseMessage) {
  const encoder = getEncoding("cl100k_base");
  const content =
    typeof message.content === "string"
      ? message.content
      : JSON.stringify(message.content);
  const token = encoder.encode(content);
  return token.length;
}

async function summaryHistoryMessage(messages) {
  try {
    const historyMessagesToText = getBufferString(messages, "用户", "AI助手");
    const summaryPrompt = `
      请你总结以下对话的核心内容，保留重要信息:
      ${historyMessagesToText}
    `;
    const response = await model.invoke([new SystemMessage(summaryPrompt)]);
    return response.content;
  } catch (error) {
    console.error(error);
  }
}

summarizeHistoryMessage();

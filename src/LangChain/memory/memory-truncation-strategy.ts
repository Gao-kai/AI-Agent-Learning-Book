/**
 * 本章节主要讨论内存管理中截断策略（truncation）
 * 截断策略是指在内存中存储消息时，当消息数量超过指定阈值时，如何处理。
 * 例如，可以截断旧的消息，只保留最近的消息。
 */

import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  trimMessages,
} from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { getEncoding, getEncodingNameForModel } from "js-tiktoken";
import "dotenv/config";

const llm = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

/**
 * 截断策略的实现：根据消息数量截断
 */
async function truncationByMessageCount() {
  const inMemoryChatMessageHistory = new InMemoryChatMessageHistory();
  const maxMessageCount = 6;
  const messageList = [
    { type: "human", content: "我叫李四" },
    { type: "ai", content: "你好李四，很高兴认识你！" },
    { type: "human", content: "我是一名设计师" },
    {
      type: "ai",
      content: "设计师是个很有创造力的职业！你主要做什么类型的设计？",
    },
    { type: "human", content: "我喜欢艺术和音乐" },
    { type: "ai", content: "艺术和音乐都是很好的爱好，它们能激发创作灵感。" },
    { type: "human", content: "我擅长 UI/UX 设计" },
    { type: "ai", content: "UI/UX 设计非常重要，好的用户体验能让产品更成功！" },
  ];

  for (const message of messageList) {
    if (message.type === "human") {
      await inMemoryChatMessageHistory.addMessage(
        new HumanMessage(message.content),
      );
    } else if (message.type === "ai") {
      await inMemoryChatMessageHistory.addMessage(
        new AIMessage(message.content),
      );
    }
  }

  let totalMessages = await inMemoryChatMessageHistory.getMessages();
  const trimmedMessage = totalMessages.slice(-maxMessageCount);

  console.log("✅保留的消息数量:", trimmedMessage.length);
  console.log(
    "✅保留的消息内容:\n",
    trimmedMessage
      .map((msg) => `${msg.constructor.name}:${msg.content}`)
      .join("\n"),
  );
}

/**
 * 截断策略的实现：根据对话消息中消耗的Token数量截断
 */
async function truncationByTokenCount() {
  const inMemoryChatMessageHistory = new InMemoryChatMessageHistory();
  const messageList = [
    { type: "human", content: "我叫李四" },
    { type: "ai", content: "你好李四，很高兴认识你！" },
    { type: "human", content: "我是一名设计师" },
    {
      type: "ai",
      content: "设计师是个很有创造力的职业！你主要做什么类型的设计？",
    },
    { type: "human", content: "我喜欢艺术和音乐" },
    { type: "ai", content: "艺术和音乐都是很好的爱好，它们能激发创作灵感。" },
    { type: "human", content: "我擅长 UI/UX 设计" },
    { type: "ai", content: "UI/UX 设计非常重要，好的用户体验能让产品更成功！" },
  ];

  for (const message of messageList) {
    if (message.type === "human") {
      await inMemoryChatMessageHistory.addMessage(
        new HumanMessage(message.content),
      );
    } else if (message.type === "ai") {
      await inMemoryChatMessageHistory.addMessage(
        new AIMessage(message.content),
      );
    }
  }

  let totalMessages = await inMemoryChatMessageHistory.getMessages();
  /**
   * trimMessages可以按照消息数量或者token总数对历史消息数组进行截断
   */
  let trimmedMessage = await trimMessages(totalMessages, {
    maxTokens: 120,
    strategy: "last",
    tokenCounter: (msgs) => countTokens(msgs),
    includeSystem: true,
  });

  const encodingInstance = getEncoding("cl100k_base");
  console.log(`✅总 token 数: ${countTokens(trimmedMessage)}/${120}`);
  console.log("✅保留的消息数量:", trimmedMessage.length);
  console.log(
    "✅保留的消息内容:\n",
    trimmedMessage
      .map((msg) => {
        const content =
          typeof msg.content === "string"
            ? msg.content
            : JSON.stringify(msg.content);
        const tokens = encodingInstance.encode(content).length;
        return `${msg.constructor.name} (${tokens} tokens): ${content}`;
      })
      .join("\n"),
  );
}

// truncationByMessageCount();
truncationByTokenCount();

/**
 * 自定义Token计算器
 */
function countTokens(messages: BaseMessage[]) {
  // 根据模型名称获取对应的编码名称
  const encodingName = getEncodingNameForModel("gpt-4");
  const encodingInstance = getEncoding(encodingName);
  let totalTokens = 0;
  for (const message of messages) {
    let content =
      typeof message.content === "string"
        ? message.content
        : JSON.stringify(message.content);
    let msgToken = encodingInstance.encode(content).length;
    totalTokens += msgToken;
  }
  return totalTokens;
}

/**
 * 今日目标
 * - 学完Memory的三种策略
 * - 阅读langchain官方文档短期记忆
 * - 阅读langchain官方文档长期记忆
 *
 * LangChain 内存管理
 * 1. 为什么要使用
 * 2. 应用层面如何使用
 * 3. 底层实现原理的代码展示
 *
 * ## 问题1: 通过手动构建MessageList的方式的缺点
 * 大模型的上下文大小总是有限的，因此当无限往memory中增加message的时候最终总会超出限制
 *
 * 所以：如何解决大模型中的Memory管理呢
 * 1. 截断
 * 保留最近的几条message将之前的舍弃
 * 实现方式为根据总Token数量来保留最近的message
 *
 * 2. 总结
 * 将之前的所有上下文信息做一个总结，保存这个总结和最近的几条message
 * 表现为cursor等工具达到上下文限制自动触发总结
 * 实现方式为调用大模型对之前的message做一个总结
 *
 * 3. 检索
 * 将上下文信息存储到向量数据库，通过语义化检索找到之前的上下文信息
 *
 * ##  问题2: LangChain提供了哪些memory有关的API呢？
 *
 * 1. ChatMessageHistory 文档
 * 存储层的API 决定了message存储在那个位置
 * InMemoryChatMessageHistory 提供了新增 删除和获取方法来获取聊天历史消息
 * 这一步可以替代手动构建message数组的过程
 *
 * 2. trimMessages API
 *
 * ## 问题3:对话过程持久化
 * 也可以将对话过程持久化到数据库中，例如MySQL、PostgreSQL等数据库
 * 这样可以在不同时间点、不同设备上继续之前的对话
 * 这就是长期记忆（LTM long-term memory）
 * 存储在内存里的就是短期记忆（STM short-term memory）
 *
 *
 *
 */

import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import {
  BaseMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";

dotenv.config();

const llm = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

const chatMessageHistory = new InMemoryChatMessageHistory();

async function firstConversation() {
  console.log("======== ✅ first conversation ========");

  const systemMessage = new SystemMessage(
    "你是一个智能、高效、理性的股票分析助手",
  );
  await chatMessageHistory.addMessage(systemMessage);

  const humanMessage = new HumanMessage({
    name: "user",
    content: "你好，我想请你分析一下商业航天板块的股票走势",
  });
  await chatMessageHistory.addMessage(humanMessage);

  const historyMessages = await chatMessageHistory.getMessages();

  const response = await llm.invoke([...historyMessages]);
  return response;
}

async function secondConversation() {
  console.log("======== ✅ second conversation ======");
  const humanMessage = new HumanMessage({
    name: "user",
    content: "我希望你综合分析下通宇通讯的股票走势",
  });
  await chatMessageHistory.addMessage(humanMessage);

  const historyMessages = await chatMessageHistory.getMessages();

  const response = await llm.invoke([...historyMessages]);
  return response;
}
/**
 * 练习通过 InMemoryChatMessageHistory 来管理内存
 * 1. 新增消息
 * 2. 新增一组消息
 * 3. 获取所有历史消息
 */
async function practiceMemoryAPI() {
  try {
    const response1 = await firstConversation();
    console.log("🚀 AI大模型第一次回答:", response1.content);
    await chatMessageHistory.addMessage(response1);

    const response2 = await secondConversation();
    console.log("🚀 AI大模型第二次回答:", response2.content);
    await chatMessageHistory.addMessage(response2);

    const totalMessages = await chatMessageHistory.getMessages();
    console.log("🚀 聊天历史消息长度:", totalMessages.length);
    for (const message of totalMessages) {
      console.log(
        message.type === "human" ? "用户" : "助手",
        (message.content as string).substring(0, 100),
      );
    }
  } catch (error) {
    console.error("❌ 任务失败:", error);
  }
}

practiceMemoryAPI();

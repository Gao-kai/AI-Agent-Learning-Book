import "dotenv/config";
import path from "path";
import { ChatOpenAI } from "@langchain/openai";
import { FileSystemChatMessageHistory } from "@langchain/community/stores/message/file_system";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { fileURLToPath } from "url";

const llm = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0.1,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

const currFilePath = fileURLToPath(import.meta.url);
const __dirname = path.dirname(currFilePath);
const chatMessagesFilePath = path.join(__dirname, "./chat_history.json");
console.log("📃文件保存路径：", chatMessagesFilePath);

const fileSystemChatMessageHistory = new FileSystemChatMessageHistory({
  sessionId: "booker_gao",
  userId: "booker_gao",
  filePath: chatMessagesFilePath,
});

async function firstConversation() {
  console.log("======== ✅ first conversation ========");

  const systemMessage = new SystemMessage(
    "你是一个智能、高效、理性的股票分析助手",
  );
  await fileSystemChatMessageHistory.addMessage(systemMessage);

  const humanMessage = new HumanMessage({
    name: "user",
    content: "你好，我想请你分析一下商业航天板块的股票走势",
  });
  await fileSystemChatMessageHistory.addMessage(humanMessage);

  const historyMessages = await fileSystemChatMessageHistory.getMessages();

  const response = await llm.invoke([...historyMessages]);
  return response;
}

async function secondConversation() {
  console.log("======== ✅ second conversation ======");
  const humanMessage = new HumanMessage({
    name: "user",
    content: "我希望你综合分析下通宇通讯的股票走势",
  });
  await fileSystemChatMessageHistory.addMessage(humanMessage);

  const historyMessages = await fileSystemChatMessageHistory.getMessages();

  const response = await llm.invoke([...historyMessages]);
  return response;
}
/**
 * 练习通过 FileSystemChatMessageHistory 来管理内存
 * 因为其继承了 ChatMessageHistory 类，所以可以使用其提供的方法来管理内存
 * 1. 新增消息
 * 2. 新增一组消息
 * 3. 获取所有历史消息
 */
async function practiceFileSystemMemoryAPI() {
  try {
    const response1 = await firstConversation();
    console.log("🚀 AI大模型第一次回答:", response1.content);
    await fileSystemChatMessageHistory.addMessage(response1);

    const response2 = await secondConversation();
    console.log("🚀 AI大模型第二次回答:", response2.content);
    await fileSystemChatMessageHistory.addMessage(response2);

    const totalMessages = await fileSystemChatMessageHistory.getMessages();
    console.log("🚀 聊天历史消息长度:", totalMessages.length);
    for (const message of totalMessages) {
      console.log(
        message.type === "human" ? "用户" : "助手",
        (message.content as string).substring(0, 50),
      );
    }
  } catch (error) {
    console.error("❌ 任务失败:", error);
  }
}

// practiceFileSystemMemoryAPI();

async function restoreChatHistory() {
  try {
    const restoreChatMessageHistory = new FileSystemChatMessageHistory({
      sessionId: "booker_gao",
      userId: "booker_gao",
      filePath: chatMessagesFilePath,
    });
    const restoredMessages = await restoreChatMessageHistory.getMessages();
    console.log("🚀 恢复的聊天历史消息长度:", restoredMessages.length);
    console.log("🚀 在原来聊天历史消息上继续对话");

    const humanMessage = new HumanMessage({
      name: "user",
      content: "好的，帮我分析下这家公司在卫星通信领域的具体布局",
    });
    await restoreChatMessageHistory.addMessage(humanMessage);

    const totalMessages = await restoreChatMessageHistory.getMessages();
    const response = await llm.invoke([...totalMessages]);
    console.log("🚀 AI大模型第三次回答:", response.content);
  } catch (error) {
    console.error("❌ 恢复聊天历史消息失败:", error);
  }
}

restoreChatHistory();

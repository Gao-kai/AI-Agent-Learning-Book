/**
 * 本章节主要讨论 通过将聊天历史记录存入向量数据库
 * 通过向量数据库来实现内存管理和长记忆存储的方案
 */
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import {
  DataType,
  IndexType,
  MetricType,
  MilvusClient,
} from "@zilliz/milvus2-sdk-node";
import "dotenv/config";

const VICTOR_DIMENSIONS = 1024;
const COLLECTION_NAME = "chat_history";

const embeddingModel = new OpenAIEmbeddings({
  model: process.env.EMBEDDING_MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
  dimensions: VICTOR_DIMENSIONS,
});

const llm = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

/**
 * 通过Node连接数据库
 */
async function createConnection() {
  try {
    const client = new MilvusClient({
      address: process.env.MILVUS_ADDRESS,
    });
    await client.connectPromise;
    console.log("✅ Milvus向量数据库连接成功!");
    return client;
  } catch (error) {
    console.error(error);
  }
}

/**
 * 创建集合
 */
async function createCollection(client: MilvusClient) {
  console.log("创建集合...");
  await client.createCollection({
    collection_name: COLLECTION_NAME,
    fields: [
      {
        name: "id",
        data_type: DataType.VarChar,
        max_length: 50,
        is_primary_key: true,
      },
      {
        name: "vector",
        data_type: DataType.FloatVector,
        dim: VICTOR_DIMENSIONS,
      },
      { name: "content", data_type: DataType.VarChar, max_length: 5000 },
      { name: "round", data_type: DataType.Int64 },
      { name: "timestamp", data_type: DataType.VarChar, max_length: 100 },
    ],
  });
  console.log("✓ 集合已创建");
}

/**
 * 创建索引
 */
async function createIndex(client: MilvusClient) {
  console.log("\n创建索引...");
  await client.createIndex({
    collection_name: COLLECTION_NAME,
    field_name: "vector",
    index_type: IndexType.IVF_FLAT,
    metric_type: MetricType.COSINE,
  });
  console.log("✓ 索引已创建");
}

/**
 * 加载集合
 * Milvus的架构决定了每次操作数据之前都必须从磁盘中将数据加载到内存中
 * - 写入数据 → 数据持久化到磁盘（segment 文件）
 * - 加载集合 → 将数据从磁盘加载到内存
 * - 搜索/查询 → 在内存中进行
 *
 * - Milvus 是 面向向量搜索优化 的数据库
 * - 向量数据通常很大（几GB到几百GB）
 * - 内存是有限的 ，不可能把所有数据都加载到内存
 * - 用户可以选择性加载/卸载集合，灵活管理内存
 */
async function loadCollection(client: MilvusClient) {
  await client.loadCollection({
    collection_name: COLLECTION_NAME,
  });
  console.log(`✅ 加载数据集合 ${COLLECTION_NAME} 成功！`);
}

/**
 * 获取文本的向量嵌入
 */
async function getEmbedding(text) {
  const result = embeddingModel.embedQuery(text);
  return result;
}

/**
 * 历史消息内容向量化之后插入到数据库中
 */
async function insertChatMessage(client: MilvusClient) {
  const chatMessages = [
    {
      id: "conv_001",
      content:
        "用户: 我叫赵六，是一名数据科学家\n助手: 很高兴认识你，赵六！数据科学是一个很有趣的领域。",
      round: 1,
      timestamp: new Date().toISOString(),
    },
    {
      id: "conv_002",
      content:
        "用户: 我最近在研究机器学习算法\n助手: 机器学习确实很有意思，你在研究哪些算法呢？",
      round: 2,
      timestamp: new Date().toISOString(),
    },
    {
      id: "conv_003",
      content:
        "用户: 我喜欢打篮球和看电影\n助手: 运动和文化娱乐都是很好的爱好！",
      round: 3,
      timestamp: new Date().toISOString(),
    },
    {
      id: "conv_004",
      content: "用户: 我周末经常去电影院\n助手: 看电影是很好的放松方式。",
      round: 4,
      timestamp: new Date().toISOString(),
    },
    {
      id: "conv_005",
      content:
        "用户: 我的职业是软件工程师\n助手: 软件工程师是个很有前景的职业！",
      round: 5,
      timestamp: new Date().toISOString(),
    },
  ];

  console.log("生成向量嵌入...");
  const data = await Promise.all(
    chatMessages.map(async (message) => {
      return {
        ...message,
        vector: await getEmbedding(message.content),
      };
    }),
  );
  console.log("\n插入对话数据...");

  const insertResult = await client.insert({
    collection_name: COLLECTION_NAME,
    data,
  });
  console.log(`✓ 已插入 ${insertResult.insert_cnt} 条记录\n`);

  console.log("=".repeat(60));
  console.log("说明：已成功将对话数据插入到 Milvus 向量数据库");
  console.log("这些对话数据将用于后续的 RAG 检索");
  console.log("=".repeat(60) + "\n");
}

/**
 * 1. 基于用户提问从数据库中检索出最相关的历史消息
 * 2. 生成查询向量
 * 3. 检索最相关的2条余弦相似度
 */
async function retrievalHistoryMessage(
  client: MilvusClient,
  input: string,
  k: number,
) {
  try {
    const queryVector = await getEmbedding(input);
    const searchResult = await client.search({
      collection_name: COLLECTION_NAME,
      vector: queryVector,
      limit: k,
      metric_type: MetricType.COSINE,
      output_fields: ["id", "content", "round", "timestamp"],
    });
    console.log("✅ 向量数据库检索成功", searchResult.results);
    return searchResult.results;
  } catch (error) {
    console.log("❌ 向量数据库检索失败", error);
    return [];
  }
}

/**
 * 1. 创建历史消息存储
 * 2. 创建用户新的提示词（包含对历史消息的提问）
 * 3. 基于用户提示词对历史消息中最相关对话进行检索
 * 4. 提取检索结果的关键信息和用户输入进行组合 形成新的上下文信息
 * 5. 调用大模型进行回答
 * 5. 将新的用户输入和大模型回答做为新的一轮会话记录加入向量数据库存储
 */
async function retrievalMemory(client: MilvusClient) {
  const history = new InMemoryChatMessageHistory();
  const userInput = "我之前提到的机器学习的进度怎么样了?";

  const searchResult = await retrievalHistoryMessage(client, userInput, 2);
  let relevantHistory = "";
  if (searchResult.length > 0) {
    relevantHistory = searchResult
      .map((res, index) => {
        const { content, round } = res;
        return `
          历史会话：${index + 1}
          轮次：${round}
          消息内容: ${content}
        `;
      })
      .join("\n\n =========== \n\n");
  } else {
    console.log("⚠️ 未找到相关历史会话");
  }

  const userMessage = new HumanMessage(userInput);
  const contextMessages = relevantHistory
    ? [new HumanMessage(relevantHistory), userMessage]
    : [userMessage];

  console.log("❤️调用AI大模型对历史消息提问进行回答");
  const response = await llm.invoke(contextMessages);
  console.log("✅ 大模型回答：\n", response.content);

  await history.addMessage(userMessage);
  await history.addMessage(response);

  console.log("❤️写入新的对话到向量数据库");
  const conversationText = `用户: ${userInput}\n助手: ${response.content}`;
  const { data: queryAllData } = await client.query({
    collection_name: COLLECTION_NAME,
    filter: "",
  });

  await client.insert({
    collection_name: COLLECTION_NAME,
    data: [
      {
        id: `conv_${Date.now()}`,
        content: conversationText,
        round: queryAllData.length + 1,
        vector: await getEmbedding(conversationText),
        timestamp: new Date().toISOString(),
      },
    ],
  });
  console.log(`💾 已保存到 Milvus 向量数据库`);
}

async function main() {
  const client = await createConnection();
  await createCollection(client);
  await createIndex(client);
  await loadCollection(client);
  // await insertChatMessage(client);
  await retrievalMemory(client);
}

main();

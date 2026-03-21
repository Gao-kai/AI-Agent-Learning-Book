import { OpenAIEmbeddings } from "@langchain/openai";
import {
  DataType,
  IndexType,
  MetricType,
  MilvusClient,
} from "@zilliz/milvus2-sdk-node";
import "dotenv/config";

const MILVUS_COLLECTION_NAME = "ai_diary";
const MILVUS_VECTOR_DIM = 1024;

/**
 * 获取文本的嵌入向量
 * @param text 要嵌入的文本
 * @returns 文本的嵌入向量
 */
async function getEmbedding(text: string) {
  return await embeddingModel.embedQuery(text);
}

/**
 * 格式化数据，为每个数据项添加嵌入向量
 * @param data 要格式化的数据项数组
 * @returns 格式化后的数据项数组，每个数据项包含原始数据和对应的嵌入向量
 */
async function formatDataByEmbedding(data: Record<string, any>[]) {
  try {
    const results = await Promise.all(
      data.map(async (item) => {
        return {
          ...item,
          vector: await getEmbedding(item.content),
        };
      }),
    );
    return results;
  } catch (error) {
    console.error(`Error formatting data by embedding: ${error}`);
    return [];
  }
}

const embeddingModel = new OpenAIEmbeddings({
  model: process.env.EMBEDDING_MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
  dimensions: MILVUS_VECTOR_DIM,
});

const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS,
});
console.log("Connecting to Milvus...");
await client.connectPromise;
console.log("Connected to Milvus!\n");

const diaryContents = [
  {
    id: "diary_001",
    content:
      "今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。",
    date: "2026-01-10",
    mood: "happy",
    tags: ["生活", "散步"],
  },
  {
    id: "diary_002",
    content:
      "今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。",
    date: "2026-01-11",
    mood: "excited",
    tags: ["工作", "成就"],
  },
  {
    id: "diary_003",
    content: "周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。",
    date: "2026-01-12",
    mood: "relaxed",
    tags: ["户外", "朋友"],
  },
  {
    id: "diary_004",
    content:
      "今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。",
    date: "2026-01-12",
    mood: "curious",
    tags: ["学习", "技术"],
  },
  {
    id: "diary_005",
    content:
      "晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。",
    date: "2026-01-13",
    mood: "proud",
    tags: ["美食", "家庭"],
  },
];

async function insertData(data: Record<string, any>[]) {
  try {
    /**
     * Create a collection if it doesn't exist
     */
    await client.createCollection({
      collection_name: MILVUS_COLLECTION_NAME,
      fields: [
        {
          name: "id",
          data_type: DataType.VarChar,
          is_primary_key: true,
          max_length: 50,
        },
        {
          name: "vector",
          data_type: DataType.FloatVector,
          dim: MILVUS_VECTOR_DIM,
        },
        {
          name: "content",
          data_type: DataType.VarChar,
          max_length: 5000,
        },
        {
          name: "date",
          data_type: DataType.VarChar,
          max_length: 50,
        },
        {
          name: "mood",
          data_type: DataType.VarChar,
          max_length: 50,
        },
        {
          name: "tags",
          data_type: DataType.Array,
          element_type: DataType.VarChar,
          max_capacity: 10,
          max_length: 50,
        },
      ],
    });
    console.log(`Collection ${MILVUS_COLLECTION_NAME} created successfully!\n`);

    /**
     * Create an index if it doesn't exist
     */
    await client.createIndex({
      collection_name: MILVUS_COLLECTION_NAME,
      field_name: "vector",
      index_type: IndexType.IVF_FLAT,
      metric_type: MetricType.COSINE,
      params: {
        nlist: 1024,
      },
    });
    console.log(`Index vector_index created successfully!\n`);

    /**
     * Load the collection into memory
     */
    await client.loadCollection({
      collection_name: MILVUS_COLLECTION_NAME,
    });
    console.log(`Collection ${MILVUS_COLLECTION_NAME} loaded successfully!\n`);

    /**
     * Insert data into the collection
     */
    const insertResult = await client.insert({
      collection_name: MILVUS_COLLECTION_NAME,
      data: await formatDataByEmbedding(data),
    });
    console.log(
      `Data inserted successfully! Insert result: ${JSON.stringify(insertResult)}\n`,
    );
  } catch (error) {
    console.error(`Error inserting data: ${error}`);
  }
}

async function queryData(query: string) {
  try {
    const queryVector = await getEmbedding(query);
    const queryResult = await client.search({
      collection_name: MILVUS_COLLECTION_NAME,
      vector: queryVector,
      limit: 2,
      metric_type: MetricType.COSINE,
      output_fields: ["id", "content", "date", "mood", "tags"],
    });
    console.log(`Found ${queryResult.results.length} results:\n`);
    queryResult.results.forEach((result, index) => {
      console.log(`${index + 1}. [Score: ${result.score.toFixed(4)}]`);
      console.log(`   ID: ${result.id}`);
      console.log(`   Date: ${result.date}`);
      console.log(`   Mood: ${result.mood}`);
      console.log(`   Tags: ${result.tags?.join(", ")}`);
      console.log(`   Content: ${result.content}\n`);
    });
  } catch (error) {
    console.error(`Error querying data: ${error}`);
  }
}

async function updateData(id: string, updateContent: Record<string, any>) {
  try {
    /**
     * Generate embedding vector for the updated content
     */
    const vector = await getEmbedding(updateContent.content);
    const updateData = {
      ...updateContent,
      vector,
    };
    /**
     * Update data into the collection
     */
    const updateResult = await client.upsert({
      collection_name: MILVUS_COLLECTION_NAME,
      data: [
        {
          id,
          ...updateData,
        },
      ],
    });
    console.log(
      `Data updated successfully! Update result: ${JSON.stringify(updateResult)}\n`,
    );
  } catch (error) {
    console.error(`Error updating data: ${error}`);
  }
}

async function deleteData(id: string) {
  try {
    /**
     * Delete data from the collection
     */
    const deleteResult = await client.delete({
      collection_name: MILVUS_COLLECTION_NAME,
      filter: `id == "${id}"`, // 支持条件删除 批量删除 id in ["diary_004", "diary_005"]
    });
    console.log(
      `Data deleted successfully! Delete result: ${JSON.stringify(deleteResult)}\n`,
    );
  } catch (error) {
    console.error(`Error deleting data: ${error}`);
  }
}
// insertData(diaryContents);
// queryData("我想看看户外活动的日记内容");
// queryData("我想看看学习相关的日记内容");
// updateData("diary_004", {
//   content:
//     "今天学习了 MySQL 数据库和Milvus向量数据库，感觉很有意思。搜索技术真的很强大。",
//   date: "2026-01-10",
//   mood: "sad",
//   tags: ["生活", "散步", "朋友"],
// });
deleteData("diary_005");

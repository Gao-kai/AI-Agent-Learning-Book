/**
 * output parser 示例 07 流式输出如何选择
 *
 * 为什么structuredModel方案不可以？
 * 使用structuredModel虽然可以简化StructuredOutputParser和tool的工具调用时的输出格式化问题
 * 但是无法满足流式输出的场景，因为模型会将返回的content先转化为json进行schema的校验后当作一个chunk返回
 * 
 * 用什么方案？
 * 1. 使用StructuredOutputParser的json-parser方案 将返回的chunk逐行打印
 * 2. 使用tool_calls的方案
 
 */
import z from "zod";
import { ChatOpenAI } from "@langchain/openai";
import "dotenv/config";
import { StructuredOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
  timeout: 10000,
  maxRetries: null,
});

const playerZodSchema = z.object({
  name: z.string().describe("姓名").optional(),
  birth_date: z.string().describe("出生日期").optional(),
  height: z.string().describe("身高").optional(),
  weight: z.string().describe("体重").optional(),
  achievements: z.string().describe("主要成就").optional(),
  records: z.string().describe("重要记录").optional(),
});

const modelWithTools = model.bindTools([
  {
    name: "extract_nba_player_info",
    description: "提取名人的个人和职业生涯的信息,然后转化为结构化数据",
    schema: playerZodSchema,
  },
]);

const parser = StructuredOutputParser.fromZodSchema(playerZodSchema);
const prompt = `介绍篮球运动员布里奇斯`;

try {
  console.log("大模型开始调用");
  const stream = await modelWithTools.stream(prompt);
  console.log("📡 接收流式数据:\n");
  let chunksCount = 0;
  for await (const chunk of stream) {
    chunksCount++;
    console.log(chunk.tool_call_chunks);
    if (chunk.tool_call_chunks.length > 0) {
      process.stdout.write(chunk.tool_call_chunks[0].args);
    }
  }

  console.log("\n大模型共返回chunk个数:", chunksCount);
} catch (error) {
  console.error(error);
}

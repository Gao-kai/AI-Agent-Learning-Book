/**
 * output parser 示例 05 使用工具调用解析结构化数据
 */
import z from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import "dotenv/config";

const playerZodSchema = z.object({
  name: z.string().describe("姓名"),
  birth_date: z.string().describe("出生日期"),
  height: z.string().describe("身高"),
  weight: z.string().describe("体重"),
  achievements: z.string().describe("主要成就"),
  records: z.string().describe("重要记录"),
});

const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
  timeout: undefined,
  maxTokens: undefined,
  maxRetries: undefined,
});

/**
 * questions:
 * 1. 直接绑定一个没有工具调用函数的工具用来获取工具调用时的结构化参数是否可行？
 * 2. 为什么说这种方式比 output parser 更好？
 */
const modelWithTools = model.bindTools([
  {
    name: "extract_nba_player_info",
    description: "提取NBA球员的个人和职业生涯的信息,然后转化为结构化数据",
    schema: playerZodSchema,
  },
]);
const prompt = `介绍勒布朗詹姆斯的个人信息和职业生涯的信息。`;

try {
  console.log("大模型开始调用");
  const response = await modelWithTools.invoke(prompt);
  console.log("大模型返回的工具调用数据:", response.tool_calls);
  const toolCall = response.tool_calls[0];
  console.log("工具调用参数:", toolCall.args);
} catch (error) {}

/**
 * output parser 示例 04
 * 需求：使用LangChain核心组件StructuredOutputParser解析大模型返回的原始数据为结构化格式的数据
 *
 */
import z from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import "dotenv/config";

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

const playerZodSchema = z.object({
  name: z.string().describe("姓名"),
  birth_date: z.string().describe("出生日期"),
  height: z.string().describe("身高"),
  weight: z.string().describe("体重"),
  achievements: z.string().describe("主要成就"),
  records: z.string().describe("重要记录"),
});

const parser = StructuredOutputParser.fromZodSchema(playerZodSchema);

const prompt = `
    帮我介绍勒布朗詹姆斯的个人信息和职业生涯的信息。
    ${parser.getFormatInstructions()}
`;

console.log("发送给大模型的提示词是:\n", prompt);

try {
  console.log("大模型开始调用");
  const response = await model.invoke(prompt);
  console.log("大模型返回的原始数据:", response.content);
  const parsedContent = await parser.parse(response.content as string);
  console.log("格式化后的JSON数据:\n", parsedContent);
} catch (error) {}

/**
 * output parser 示例 03
 * 需求：使用LangChain核心组件StructuredOutputParser解析大模型返回的原始数据为结构化格式的数据
 *
 * 总结：
 * 1. 引入StructuredOutputParser组件中的fromNamesAndDescriptions方法来实例化一个parser对象
 * 2. fromNamesAndDescriptions的意思是可以直接通过字段和自然语言来描述希望大模型返回的原始数据的结构
 * 3. 在提示词中调用parser.getFormatInstructions方法获取格式化指令,本质就是将如何解析结构化数据的规则加入到提示词告诉大模型
 * Your output will be parsed and type-checked according to the provided schema instance,
 * so make sure all fields in your output match the schema exactly and there are no trailing commas!
 * Here is the JSON Schema instance your output must adhere to. Include the enclosing markdown codeblock:
 * ```json
 * {
 *  "type":"object",
 *  "properties":{"name":{"type":"string","description":"姓名"},"birth_date":{"type":"string","description":"出生日期"},"height":{"type":"string","description":"身高"},"weight":{"type":"string","description":"体重"},"achievements":{"type":"string","description":"主要成就"},"records":{"type":"string","description":"重要记录"}},"required":["name","birth_date","height","weight","achievements","records"],"additionalProperties":false,"$schema":"http://json-schema.org/draft-07/schema#"
 * }
 * 4. 大模型返回的原始数据依然是包含markdown格式的json数据，但是parser会自动将其转换为可用的结构化JSON格式
 */
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

const parser = StructuredOutputParser.fromNamesAndDescriptions({
  name: "姓名",
  birth_date: "出生日期",
  height: "身高",
  weight: "体重",
  achievements: "主要成就",
  records: "重要记录",
});

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

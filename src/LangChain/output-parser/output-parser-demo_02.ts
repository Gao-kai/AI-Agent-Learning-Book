/**
 * output parser 示例 02
 * 需求：使用LangChain核心组件OutputParser解析大模型返回的原始数据为JSON格式的数据
 *
 * 总结：
 * 1. 引入JsonOutputParser组件并实例化parser
 * 2. 调用parser.getFormatInstructions方法获取格式化指令,本质就是将如何解析JSON的规则加入到提示词告诉大模型
 * 3. 调用parser.parse方法解析大模型返回的原始数据，虽然大模型返回的数据依然包含Markdown的格式，但是parser会自动将其转换为JSON格式
 * 4. 打印解析后的JSON数据
 */

import { ChatOpenAI } from "@langchain/openai";
import { JsonOutputParser } from "@langchain/core/output_parsers";
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

const parser = new JsonOutputParser();

const prompt = `
    帮我介绍勒布朗詹姆斯的个人信息，其中需要如下字段信息，并且要求以JSON格式返回：
    1. 姓名name
    2. 出生日期birth_date
    3. 身高height
    4. 体重weight
    5. 主要成就achievements
    6. 重要记录records

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

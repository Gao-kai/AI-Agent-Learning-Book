/**
 * output parser 示例 01
 * 需求：如何要求大模型返回给你的数据是一个JSON格式?
 *
 * 解答：
 * 1. 首先在提示词prompt中添加需要返回JSON格式的的指令
 * 2. 在调用大模型的时候会自动将大模型返回的原始数据解析为JSON格式的数据
 *
 * 问题：
 * 1. 大模型虽然返回的是解析后JSON格式的数据，但是会带有Markdown的格式，比如```json这种格式
 * 2. 直接使用JSON.parse()方法解析大模型返回的原始数据，会报错,需要用户手动替换多余格式
 *
 * 进阶：不需要用户主动替换多余格式，大模型会自动解析为JSON格式的数据
 */

import { ChatOpenAI } from "@langchain/openai";
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

const prompt = `
    帮我介绍勒布朗詹姆斯的个人信息，其中需要如下字段信息，并且要求以JSON格式返回：
    1. 姓名
    3. 出生日期
    4. 身高
    5. 体重
    6. 主要成就
    7. 重要记录
`;

const response = await model.invoke(prompt);

console.log("大模型返回的原始数据:", response.content);

console.log("格式化后的JSON数据:", JSON.parse(response.content as string));

/**
 * 
大模型返回的原始数据: ```json
{
  "姓名": "勒布朗·詹姆斯",
  "出生日期": "1984年12月30日",
  "身高": "203厘米",
  "体重": "113公斤",
  "主要成就": [
    "4次NBA总冠军",
    "4次NBA总决赛MVP",
    "20次NBA全明星",
    "2次奥运会金牌",
    "NBA历史得分王",
    "NBA历史出场次数最多球员"
  ],
  "重要记录": [
    "NBA历史总得分最高纪录保持者",
    "NBA历史上最年轻的30000分先生",
    "NBA历史上唯一一位在三个不同球队获得总冠军的球员",
    "NBA历史上唯一一位在三个不同联盟（东部、西部）都获得过总冠军的球员",
    "NBA历史上唯一一位在20岁到38岁期间都获得过总冠军的球员"
  ]
}
```
 */

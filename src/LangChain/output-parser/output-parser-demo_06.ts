/**
 * output parser 示例 06 使用model.withStructuredOutput方法
 * model.withStructuredOutput方法会自己判断模型是否支持tool_call:
 * 1. 如果命中工具调用，则直接生成工具需要的结构化参数后传递给工具
 * 2. 如果没有命中工具调用，则使用StructuredOutputParser.fromZodSchema来转换结构化参数
 * 
 * 模型返回将直接是结构化后的数据，比如：
 * {
    name: "LeBron James",
    birth_date: "1984-12-30",
    height: "206 cm",
    weight: "113 kg",
    achievements: "4次NBA总冠军,4次总决赛最有价值球员,4次常规赛最有价值球员,20次全明星,2004年最佳新秀,NBA历史得分王,3届奥运金牌",
    records: "NBA历史总得分王,NBA历史上唯一一位达成40000分+10000篮板+10000助攻的球员,NBA出战场次与时间最多纪录保持者之一,季后赛总得分纪录保持者,入选全明星次数最多纪录保持者之一,连续得分上双场次纪录保持者,从1000分到40000分每个千分里程碑均保持最年轻纪录",
  }
 */
import z from "zod";
import { ChatOpenAI } from "@langchain/openai";
import "dotenv/config";
import { log } from "console";

const playerZodSchema = z.object({
  name: z.string().describe("姓名").optional(),
  birth_date: z.string().describe("出生日期").optional(),
  height: z.string().describe("身高").optional(),
  weight: z.string().describe("体重").optional(),
  achievements: z.string().describe("主要成就").optional(),
  records: z.string().describe("重要记录").optional(),
});

const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
  timeout: 60000, // 60秒超时
  maxRetries: 2,
});

const structuredModel = model.withStructuredOutput(playerZodSchema);

const modelWithTools = model.bindTools([
  {
    name: "extract_nba_player_info",
    description: "提取NBA球员的个人和职业生涯的信息,然后转化为结构化数据",
    schema: playerZodSchema,
  },
]);

const prompt = `介绍勒布朗詹姆斯的个人信息和职业生涯的信息。以Json格式返回即可`;

try {
  console.log("大模型开始调用");
  const response = await structuredModel.invoke(prompt);
  console.log("大模型返回的结果:", response);
} catch (error) {
  log(error);
}

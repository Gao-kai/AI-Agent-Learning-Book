import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
dotenv.config({ quiet: true });

const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  temperature: 0,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

const response = await model.invoke("请你介绍一下自己");
console.log(response.content);

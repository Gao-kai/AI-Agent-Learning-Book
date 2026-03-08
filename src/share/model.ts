import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import "dotenv/config";

const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
  // timeout: undefined,
  // maxTokens: undefined,
  // maxRetries: undefined,
});

const embeddingModel = new OpenAIEmbeddings({
  model: process.env.EMBEDDING_MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

export { model, embeddingModel };

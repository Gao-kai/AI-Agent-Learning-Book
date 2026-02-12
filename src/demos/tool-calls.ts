import { DynamicStructuredTool, tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import fs from "node:fs/promises";
import "dotenv/config";
import {
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import writeFile from "../tools/write-file";

/**
 * 1. 创建对话模型
 */
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

/**
 * 2. 创建工具
 */
const readFile = tool(
  async function ({ filePath }) {
    console.log("filePath", filePath);
    const content = await fs.readFile(filePath, "utf-8");
    console.log("文件读取成功");
    return `读取到的文件内容为\n${content}`;
  },
  {
    name: "read_file",
    description: "根据提供的文件路径（绝对路径或相对路径），读取文件内容",
    schema: z.object({
      filePath: z.string().describe("文件路径"),
    }),
  },
);

/**
 * 3. 模型绑定工具
 */
const tools = [readFile, writeFile];
const modelWithTools = model.bindTools(tools);

/**
 * 4. 模型调用
 */
const systemMessage = new SystemMessage(
  `
    你是一个编程助手，可以使用工具读取文件并解释代码

    工作流程
        - 用户要求读取文件时立即调用read_file工具
        - 等待工具读取文件内容并返回结果
        - 基于返回文件内容进行分析和解释

    可用工具列表
        - read_file
    `,
);
const humanMessage = new HumanMessage(
  `1. 请读取 src/tools/read-file.ts 文件内容后解释
   2. 将解释内容写入到路径为 src/tools/1.md文件中`,
);

const messages: BaseMessage[] = [systemMessage, humanMessage];

async function runWithTools() {
  let response = await modelWithTools.invoke(messages);

  if (response?.tool_calls?.length == 0) {
    console.log("最终结果:");
    console.log(response.content);
    return;
  }

  messages.push(response);

  const toolResults = await Promise.all(
    response.tool_calls.map(async (tool_call) => {
      try {
        const executedTool: DynamicStructuredTool = tools.find(
          (item) => item.name === tool_call.name,
        );
        if (!executedTool) {
          return `错误: 找不到工具 ${executedTool.name}`;
        }
        const res = await executedTool.invoke(tool_call.args as any);
        return res;
      } catch (error) {
        return `错误: ${error.message}`;
      }
    }),
  );

  response.tool_calls.forEach((tool_call, index) => {
    messages.push(
      new ToolMessage({
        content: toolResults[index] as string,
        tool_call_id: tool_call.id,
      }),
    );
  });

  runWithTools();
}

runWithTools();

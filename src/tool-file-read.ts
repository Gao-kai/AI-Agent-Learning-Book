import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import fs from "node:fs/promises";
import z from "zod";
import {
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import "dotenv/config";


/**
 * 创建对话模型
 */
const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.MODEL_NAME,
  temperature: 0,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

const readFileTool = tool(
  async function readFile({ filePath }) {
    try {
      const content = await fs.readFile(filePath, "utf-8");
      console.log(
        `  [工具调用] READ FILE(${filePath}) - 成功读取 ${content.length}字节`,
      );
      return `文件内容:\n\r ${content}`;
    } catch (error) {
      console.error(error);
    }
  },
  {
    name: "read-file",
    description:
      "用此工具来读取文件内容。当用户要求读取文件、查看代码、分析文件内容时，调用此工具。输入文件路径（可以是相对路径或者绝对路径）",
    schema: z.object({
      filePath: z.string().describe("要读取的文件路径"),
    }),
  },
);

const tools = [readFileTool];

let messages:BaseMessage[] = [
  new SystemMessage(`
    你是一个代码助手，可以使用工具读取文件并解释代码。

    工作流程：
        1. 用户要求读取文件时,立即使用read-file工具
        2. 等待工具返回文件内容
        3. 基于文件内容进行分析和解释
    
    可用工具：
        - read-file: 读取文件内容（使用此工具来读取文件内容）
    `),
  new HumanMessage("请读取src/tool-file-read.ts文件并解释代码"),
];

const modelWithTools = model.bindTools(tools);
let response = await modelWithTools.invoke(messages);
console.log(response.content);
// 四种message的类型
messages.push(response);
while (response.tool_calls && response.tool_calls.length > 0) {
  console.log(`检测到${response.tool_calls.length}个工具调用`);
  const toolsResults = await Promise.all(
    response.tool_calls.map(async (tool_call) => {
      const tool = tools.find((tool) => tool.name === tool_call.name);
      if (!tool) {
        return `错误：找不到工具${tool_call.name}`;
      }

      console.log(
        `执行工具：${tool_call.name} 执行参数：${JSON.stringify(tool_call.args)}`,
      );

      try {
        const result = await tool.invoke(tool_call.args as any);
        return result;
      } catch (error) {
        return `错误：${error?.message}`;
      }
    }),
  );

  response.tool_calls.forEach((tool_call, index) => {
    messages.push(
      new ToolMessage({
        content: String(toolsResults[index]),
        tool_call_id: String(tool_call.id),
      }),
    );
  });

  response = await modelWithTools.invoke(messages);
}

console.log(`最终回复:\n`);
console.log(response.content);

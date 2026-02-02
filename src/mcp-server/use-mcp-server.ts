import "dotenv/config";
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { ChatOpenAI } from "@langchain/openai";
import {
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import chalk from "chalk";
import { DynamicStructuredTool } from "@langchain/core/tools";

const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.MODEL_NAME,
  temperature: 0,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

const mcpClient = new MultiServerMCPClient({
  mcpServers: {
    "artest-mcp-server": {
      command: "tsx",
      args: [
        "/Users/artest/Project/AI-Agent-Learning-Book/src/mcp-server/query-user.ts",
      ],
    },
  },
});

const tools: DynamicStructuredTool[] = await mcpClient.getTools();
const modelWithTools = model.bindTools(tools);

async function readResource() {
  const resource = await mcpClient.listResources();
  let res = "";
  for (const [mcpServerName, mcpServerResources] of Object.entries(resource)) {
    for (const mcpServerResource of mcpServerResources) {
      const content = await mcpClient.readResource(
        mcpServerName,
        mcpServerResource.uri,
      );
      console.log(content);
      res += content[0].text;
    }
  }

  return res;
}

async function runWithAiAgent(question: string, maxIterations = 30) {
  const resourceContent = await readResource();
  const messages: BaseMessage[] = [
    new SystemMessage(resourceContent),
    new HumanMessage(question),
  ];
  for (let i = 0; i < maxIterations; i++) {
    console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));
    const response = await modelWithTools.invoke(messages);
    messages.push(response);

    if (!response.tool_calls || response.tool_calls?.length === 0) {
      console.log(`\n✨ AI 最终回复:\n${response.content}\n`);
      return response.content;
    }

    console.log(
      chalk.bgBlue(`🔍 检测到 ${response.tool_calls.length} 个工具调用`),
    );
    console.log(
      chalk.bgBlue(
        `🔍 工具调用: ${response.tool_calls.map((t) => t.name).join(", ")}`,
      ),
    );

    for (const tool_call of response.tool_calls) {
      const invokedTool: DynamicStructuredTool = tools.find(
        (item) => item.name === tool_call.name,
      );
      if (invokedTool) {
        const toolResult = await invokedTool.invoke(tool_call.args);
        messages.push(
          new ToolMessage({
            content: toolResult,
            tool_call_id: tool_call.id,
          }),
        );
      }
    }
  }

  return messages[messages.length - 1].content;
}

async function init() {
  //   await runWithAiAgent("查询用户 002 详细信息");
  await runWithAiAgent("MCP Server的使用指南是什么？");
  await mcpClient.close();
}

init();

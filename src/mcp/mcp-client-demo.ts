/**
 * 在AI Agent开发中使用代码调用MCP Server
 */
import {
  BaseMessage,
  HumanMessage,
  ToolMessage,
  SystemMessage,
} from "@langchain/core/messages";
import model from "../share/model";
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import chalk from "chalk";

/**
 * 1. 初始化MCP客户端,配置MCP Server的信息
 */
const mcpClient = new MultiServerMCPClient({
  mcpServers: {
    "artest-mcp-server": {
      command: "npx",
      args: [
        "-y",
        "tsx",
        "/Users/artest/Project/AI-Agent-Learning-Book/src/mcp/mcp-server-demo.ts",
      ],
    },
  },
});

console.log(chalk.bgBlue("[MCP Client 初始化完成]"));

/**
 * 2. 获取MCP Server注册的所有工具
 */
const tools = await mcpClient.getTools();
console.log(chalk.bgBlue("[MCP Client 获取工具完成]"));
console.log(chalk.bgBlue("[MCP Client 工具列表]"));
console.log(
  tools.map((item) => `${item.name}: ${item.description}`).join("\n"),
);

/**
 * 3. 获取MCP Server注册的静态资源
 */
const resources = await mcpClient.listResources();
console.log(chalk.bgBlue("[MCP Client 获取静态资源列表完成]"));
let resourceContent = "";
for (const [mcpServerName, mcpServerResource] of Object.entries(resources)) {
  for (const resource of mcpServerResource) {
    const content = await mcpClient.readResource(mcpServerName, resource.uri);
    console.log(chalk.bgBlue(`[MCP Client 读取静态资源 ${resource.uri}]`));
    resourceContent += content[0].text;
  }
}
let messages: BaseMessage[] = [];
const systemMessage = new SystemMessage(resourceContent);

/**
 * 3. 绑定MCP Server注册的工具到模型
 */
const modelWithTools = model.bindTools(tools);
console.log(chalk.bgBlue("[MCP Client 绑定工具完成]"));

/**
 * 4. 运行AI Agent, 调用MCP Server注册的工具
 * @param userInput
 * @param maxIterations
 * @returns
 */
async function runAgentWithTools(
  userInput: string,
  maxIterations: number = 30,
) {
  messages.push(systemMessage, new HumanMessage(userInput));
  for (let i = 0; i < maxIterations; i++) {
    let response = await modelWithTools.invoke(messages);
    messages.push(response);
    if (response.tool_calls.length === 0) {
      console.log(chalk.bgMagenta(`💥AI最终输出如下：\n${response.content}`));
      return response.content;
    }

    for (const toolCall of response.tool_calls) {
      const invokeTool = tools.find((item) => item.name === toolCall.name);
      if (!invokeTool) {
        console.log(
          chalk.red(`[MCP Client 工具调用失败] 工具 ${toolCall.name} 不存在`),
        );
        continue;
      }
      const toolResponse = await invokeTool.invoke(toolCall);
      const toolResponseMessage = new ToolMessage({
        content: toolResponse.content,
        tool_call_id: toolCall.id,
      });
      messages.push(toolResponseMessage);
    }
  }

  // 当maxIterations轮对话之后 此时输出AI大模型返回的最近一次的消息
  return messages[messages.length - 1].content;
}

await runAgentWithTools("查询用户ID为1的用户信息");
await mcpClient.close();
console.log(chalk.bgBlue("[MCP Client 关闭完成]"));

await runAgentWithTools("我该如何使用这个MCP Serve");
await mcpClient.close();
console.log(chalk.bgBlue("[MCP Client 关闭完成]"));

import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { model } from "../share/model";
import {
  BaseMessage,
  HumanMessage,
  ToolMessage,
} from "@langchain/core/messages";
import chalk from "chalk";

const mcpClient = new MultiServerMCPClient({
  mcpServers: {
    "amap-maps-streamableHTTP": {
      url: `https://mcp.amap.com/mcp?key=${process.env.AMAP_MAPS_API_KEY}`,
    },
    Filesystem: {
      command: "npx",
      args: [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/artest/Desktop",
        "/Users/artest/Project",
      ],
      env: {},
    },
    "Chrome DevTools MCP": {
      command: "npx",
      args: ["-y", "chrome-devtools-mcp@latest", "--isolated"],
      env: {},
    },
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

const tools = await mcpClient.getTools();
const modelWithTools = model.bindTools(tools);

let messages: BaseMessage[] = [];
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
  messages.push(new HumanMessage(userInput));
  for (let i = 0; i < maxIterations; i++) {
    console.log(chalk.bgCyan(`🚀 [AI Agent 第${i + 1}轮思考中......]`));
    let response = await modelWithTools.invoke(messages);
    messages.push(response);
    if (response.tool_calls.length === 0) {
      console.log(chalk.bgMagenta(`💥 AI最终输出如下：\n${response.content}`));
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
      console.log(
        chalk.bgBlue(
          `[MCP Client 工具调用成功] 成功调用工具 ${toolCall.name} 并返回结果`,
        ),
      );
      const toolResponse = await invokeTool.invoke(toolCall);
      const toolResponseMessage = new ToolMessage({
        content:
          typeof toolResponse.content === "string"
            ? toolResponse.content
            : toolResponse.content?.text || "",
        tool_call_id: toolCall.id,
      });
      messages.push(toolResponseMessage);
    }
  }

  // 当maxIterations轮对话之后 此时输出AI大模型返回的最近一次的消息
  return messages[messages.length - 1].content;
}

// await runAgentWithTools(
//   `
//   请根据以下信息进行查询,并将结果写入到src/output目录下的周末出行.md文件中
//   - 出发地：四川省成都市新都区木锦新城A区
//   - 目的地：四川省广安市岳池县
//   - 出发时间：早上7-8点
//   - 到达时间：必须在早上11点之前

//   - 交通工具：
//     - 自驾
//     - 顺风车
//     - 高铁+打车

//    - 注意：
//     - 如果从成都直达岳池县的高铁到达太晚，可以先高铁到达距离目的地最近的有高铁的城市，然后再打车过去
//     - 至少返回3种出行方案
// `,
// );
// await mcpClient.close();
// console.log(chalk.bgBlue("[MCP Client 关闭完成]"));

await runAgentWithTools("打开高德地图首页，查找北京南站附近的酒店");
// await mcpClient.close();
console.log(chalk.bgBlue("[MCP Client 关闭完成]"));

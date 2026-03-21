import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import chalk from "chalk";
import z from "zod";

/**
 * 模拟数据库
 */
const database = {
  users: [
    {
      id: 1,
      name: "artest",
      password: "123456",
      email: "artest@example.com",
    },
    {
      id: 2,
      name: "tom",
      password: "123456",
      email: "tom@example.com",
    },
    {
      id: 3,
      name: "lily",
      password: "123456",
      email: "lily@example.com",
    },
  ],
};

/**
 * 创建MCP服务器实例
 */
const mcpServer = new McpServer({
  name: "artest-mcp-server",
  version: "1.0.0",
});

const queryUserInfoByIdSchema = z.object({
  userId: z.number().describe("用户ID"),
});

export type QueryUserInfoByIdInput = z.infer<typeof queryUserInfoByIdSchema>;

/**
 * 注册MCP Server工具-查询用户信息
 */
mcpServer.registerTool(
  "query_user",
  {
    description: "根据用户名查询用户信息",
    inputSchema: queryUserInfoByIdSchema,
  },
  async function queryUserInfoById({ userId }: QueryUserInfoByIdInput) {
    console.log(
      chalk.bgRed(
        `[MCP Server 工具调用]-query_user(查询用户工具) 用户ID:${userId}`,
      ),
    );
    const user = database.users.find((item) => item.id === userId);
    if (!user) {
      console.log(
        chalk.red(`[query_user 工具调用失败] 用户ID ${userId} 不存在`),
      );
      return {
        content: [
          {
            type: "text" as const,
            text: `用户ID ${userId} 不存在`,
          },
        ],
      };
    }
    console.log(
      chalk.bgGreen(
        `[query_user 工具调用成功] 成功查询到用户信息:${JSON.stringify(user)}`,
      ),
    );
    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify(user),
        },
      ],
    };
  },
);

/**
 * 注册MCP Resource资源-查询使用文档
 */
mcpServer.registerResource(
  "use_guide",
  "docs://guide",
  {
    description: "本MCP Server使用文档",
    mimeType: "text/plain",
  },
  async function getGuideResource() {
    return {
      contents: [
        {
          uri: "docs://guide",
          mimeType: "text/plain",
          text: `
                    本MCP Server使用文档
                    1. 本MCP Server支持的工具列表:
                        - query_user: 根据用户ID查询用户信息
                    2. 本MCP Server支持的资源列表:
                        - use_guide: 本MCP Server使用文档
                    3. MCP Client通过自然语言对话时,会根据用户输入的提示词，
                       自动调用MCP Server支持的工具或资源,从而实现用户需求。
                `,
        },
      ],
    };
  },
);

/**
 * 创建MCP服务器传输通道-标准输入输出
 */
const transport = new StdioServerTransport();

/**
 * 启动MCP服务器
 */
await mcpServer.connect(transport);
console.log(chalk.bgBlue("[MCP Server 启动完成]"));

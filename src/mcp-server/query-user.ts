import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const dataBase = {
  users: {
    "001": {
      id: "001",
      name: "张三",
      email: "zhangsan@example.com",
      role: "admin",
    },
    "002": {
      id: "002",
      name: "李四",
      email: "lisibb@example.com",
      role: "user",
    },
    "003": {
      id: "003",
      name: "王五",
      email: "wangwu@example.com",
      role: "user",
    },
  },
};

const server = new McpServer({
  name: "artest-mcp-server",
  version: "1.0.0",
});

const queryUserInputSchema = z.object({
  userId: z.string().describe("用户ID，比如001、002"),
});

type QueryUserInputSchema = z.infer<typeof queryUserInputSchema>;

server.registerTool(
  "query-user",
  {
    description: `
        查询本地数据库中用户信息
        输入用户id
        返回该用户的详细信息，包含姓名、邮箱、角色
    `,
    inputSchema: queryUserInputSchema,
  },
  async ({ userId }: QueryUserInputSchema) => {
    const user = dataBase.users[userId];
    if (!user) {
      return {
        content: [
          {
            type: "text",
            text: `用户 ID ${userId} 不存在。可用的用户ID为：001、002、003`,
          },
        ],
      };
    }

    return {
      content: [
        {
          type: "text",
          text: `用户信息：\n- ID: ${user.id}\n- 姓名: ${user.name}\n- 邮箱: ${user.email}\n- 角色: ${user.role}`,
        },
      ],
    };
  },
);

server.registerResource(
  "使用指南",
  "docs://guide",
  {
    description: "MCP Server 使用文档",
    mimeType: "text/plain",
  },
  async () => {
    return {
      contents: [
        {
          uri: "docs://guide",
          mimeType: "text/plain",
          text: `MCP Server 使用指南
        功能：提供用户查询等工具。
        使用：在 Cursor 等 MCP Client 中通过自然语言对话，Cursor 会自动调用相应工具。`,
        },
      ],
    };
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);

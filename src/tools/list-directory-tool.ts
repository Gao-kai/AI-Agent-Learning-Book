import { tool } from "@langchain/core/tools";
import z from "zod";
import fs from "node:fs/promises";

const listDirectoryTool = tool(
  async ({ directoryPath }) => {
    try {
      console.log(`[工具调用] list-directory - 开始读取目录 ${directoryPath}`);
      const files = await fs.readdir(directoryPath);
      console.log(
        `[工具调用] list-directory - 读取到 ${directoryPath}路径下有${files.length}个项目`,
      );
      return `目录内容:
      ${files.map((file) => `- ${file}`).join("\n")}`;
    } catch (error) {}
  },
  {
    name: "list-directory",
    description: "列出指定目录下的所有文件和文件夹",
    schema: z.object({
      directoryPath: z.string().describe("目录路径"),
    }),
  },
);

export default listDirectoryTool;

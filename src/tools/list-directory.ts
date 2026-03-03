import { tool } from "@langchain/core/tools";
import chalk from "chalk";
import z from "zod";
import fs from "node:fs/promises";

export type ListDirectoryToolInput = z.infer<typeof listDirectorySchema>;

const listDirectorySchema = z.object({
  dirPath: z.string().describe("要列出目录的文件夹路径"),
});

async function listDirectory({ dirPath }: ListDirectoryToolInput) {
  console.log(
    chalk.bgRed(`[工具调用]-list_directory(列出目录工具) 目录路径:${dirPath}}`),
  );
  const files = await fs.readdir(dirPath);
  const fileNames = files.map((item) => `- ${item}`).join("\n");
  console.log(
    chalk.bgRed(`[list_directory 工具调用成功] 成功列出${files.length}个项目`),
  );
  return `列出${dirPath}目录下的项目如下:\n${fileNames}`;
}

const listDirectoryTool = tool(listDirectory, {
  name: "list_directory",
  description: "列出指定目录下的所有文件和文件夹",
  schema: listDirectorySchema,
});

export default listDirectoryTool;

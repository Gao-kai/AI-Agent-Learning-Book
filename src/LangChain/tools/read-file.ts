import { tool } from "@langchain/core/tools";
import fs from "node:fs/promises";
import z from "zod";
import chalk from "chalk";

const readFileToolSchema = z.object({
  filePath: z.string().describe("文件路径"),
});

export type ReadFileToolInput = z.infer<typeof readFileToolSchema>;

const readFileTool = tool(
  async function ({ filePath }: ReadFileToolInput) {
    console.log(
      chalk.bgRed(`[工具调用]-read_file(文件读取工具) 文件路径:${filePath}`),
    );
    const content = await fs.readFile(filePath, "utf-8");
    console.log(
      chalk.bgGreen(
        `[readFile 工具调用成功] 成功读取到${content.length}字节数据`,
      ),
    );
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

export default readFileTool;

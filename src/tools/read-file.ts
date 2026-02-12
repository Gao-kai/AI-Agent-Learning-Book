import { tool } from "@langchain/core/tools";
import fs from "node:fs/promises";
import z from "zod";

const readFileToolSchema = z.object({
  filePath: z.string().describe("文件路径"),
});

export type ReadFileToolInput = z.infer<typeof readFileToolSchema>;

const readFile = tool(
  async function ({ filePath }: ReadFileToolInput) {
    console.log("[readFile 工具调用]");
    console.log("filePath", filePath);
    const content = await fs.readFile(filePath, "utf-8");
    console.log("[readFile 工具调用成功]");
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

export default readFile;

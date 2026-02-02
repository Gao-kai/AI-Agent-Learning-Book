import fs from "node:fs/promises";
import { tool } from "@langchain/core/tools";
import z from "zod";
import path from "node:path";

const fileWriteSchema = z.object({
  filePath: z.string().describe("文件路径"),
  content: z.string().describe("要写入的文件路径"),
});

type FileWriteInput = z.infer<typeof fileWriteSchema>;

const writeFileTool = tool(
  async ({ filePath, content }: FileWriteInput) => {
    try {
      const dir = path.dirname(filePath);
      await fs.mkdir(dir, { recursive: true });
      console.log(`[工具调用] write-file - 开始写入: ${filePath}文件`);
      await fs.writeFile(filePath, content, "utf-8");
      console.log(`[工具调用] write-file - 成功写入: ${content.length}字节`);
      return `文件成功写入: ${filePath}`;
    } catch (error) {
      console.log(`[工具调用] write-file - 写入错误: ${error?.message}`);
    }
  },
  {
    name: "write-file",
    description: "向指定文件写入内容时，调用此工具",
    schema: fileWriteSchema,
  },
);

export default writeFileTool;

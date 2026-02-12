import { tool } from "@langchain/core/tools";
import z from "zod";
import fs from "node:fs/promises";
import path from "node:path";

const writeFileToolSchema = z.object({
  filePath: z.string().describe("文件路径"),
  content: z.string().describe("文件内容"),
});

export type WriteFileToolInput = z.infer<typeof writeFileToolSchema>;

const writeFile = tool(
  async ({ filePath, content }: WriteFileToolInput) => {
    try {
      // 确保传入的文件路径存在 如果不存在 则递归创建一个文件夹
      const dirname = path.dirname(filePath);
      fs.mkdir(dirname, { recursive: true });
      fs.writeFile(filePath, content, "utf-8");
      return `文件写入成功\n文件路径为${filePath}`;
    } catch (error) {
      console.error("文件写入失败", error.message);
    }
  },
  {
    name: "write_file",
    description: "将内容写入指定路径的文件中，如果文件路径不存在则先创建后写入",
    schema: writeFileToolSchema,
  },
);

export default writeFile;

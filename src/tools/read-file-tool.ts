import { tool } from "@langchain/core/tools";
import { z } from "zod";
import fs from "node:fs/promises";

export const fileReadSchema = z.object({
  filePath: z.string().describe("要读取的文件路径"),
});

export type FileReadInput = z.infer<typeof fileReadSchema>;

const readFileTool = tool(
  async ({ filePath }: FileReadInput) => {
    try {
      console.log(`[工具调用] read-file - 开始读取: ${filePath}文件`);
      const content = await fs.readFile(filePath, "utf-8");
      console.log(`[工具调用] read-file - 成功读取: ${content.length}字节`);
      return `文件内容:\n${content}`;
    } catch (error) {
      console.log(`[工具调用] read-file - 读取错误: ${error?.message}`);
    }
  },
  {
    name: "read-file",
    description: `
        当用户要求读取文件、查看代码、分析文件内容时，调用此工具。
        此工具接受参数为文件路径（相对路径或者绝对路径）
    `,
    schema: fileReadSchema,
  },
);

export default readFileTool;

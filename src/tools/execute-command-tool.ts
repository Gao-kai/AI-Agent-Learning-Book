import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { spawn } from "node:child_process";

const executeCommandSchema = z.object({
  command: z.string().describe("终端执行的命令"),
  workDirectionary: z.string().describe("执行命令时工作目录(CWD)").optional(),
});

export type ExecuteCommandInput = z.infer<typeof executeCommandSchema>;

const executeCommandTool = tool(
  async ({ command, workDirectionary }: ExecuteCommandInput) => {
    const cwd = workDirectionary || process.cwd();
    return new Promise((resolve, reject) => {
      console.log(`[工具调用] exec-command - 开始执行命令`);
      const [cmd, ...args] = command.split(" ");
      const child = spawn(cmd, args, {
        cwd: cwd,
        stdio: "inherit",
        shell: true,
      });

      let errorMsg = "";

      child.on("error", (error) => {
        errorMsg = error.message;
      });

      child.on("close", (code) => {
        if (code === 0) {
          console.log(`[工具调用] exec-command("${command}") - 执行成功`);
          const cwdInfo = workDirectionary
            ? `
            \n\n重要提示:
            命令在目录 "${workDirectionary}" 中执行成功。
            如果需要在这个项目目录中继续执行命令，请使用 workingDirectory: "${workDirectionary}" 参数
            不要使用 cd 命令。`
            : "";
          resolve(`命令执行成功: ${command}${cwdInfo}`);
        } else {
          console.log(
            ` [工具调用] exec-command("${command}") - 执行失败，退出码: ${code}`,
          );
          resolve(
            `命令执行失败，退出码: ${code}${errorMsg ? "\n错误: " + errorMsg : ""}`,
          );
        }
      });
    });
  },
  {
    name: "exec-command",
    description: "执行终端命令时，调用此工具",
    schema: executeCommandSchema,
  },
);

export default executeCommandTool;

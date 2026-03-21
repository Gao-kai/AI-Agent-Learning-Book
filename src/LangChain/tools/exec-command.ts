/**
 * nodejs中执行命令的API是什么？
 * spawn方法可以指定在cwd目录下执行命令，并且可以实时输出到控制台。
 * spawn方法的参数说明：
 * - command: 要执行的命令字符串，例如"ls -la"
 * - args: 命令的参数数组，例如["-la"]
 * - options: spawn选项对象，可以指定cwd、stdio、shell等选项
 *   - cwd: 指定命令执行的当前工作目录，默认为process.cwd()
 *   - stdio: 指定子进程的输入输出方式，"inherit"表示继承父进程的输入输出，可以实时输出到控制台
 *   - shell: 是否使用shell执行命令，默认为false，如果为true则可以执行复杂的shell命令
 *
 * 下面是一个使用spawn方法执行"ls -la"命令的示例代码：
 */
import { tool } from "@langchain/core/tools";
import chalk from "chalk";
import { spawn, SpawnOptions } from "node:child_process";
import z from "zod";

export type ExecCommandToolInput = z.infer<typeof execCommandToolSchema>;

async function execCommand({ command, cwd }: ExecCommandToolInput) {
  console.log(
    chalk.bgRed(
      `[工具调用]-exec_command(执行命令工具) 命令:${command}, 工作目录:${cwd || process.cwd()}`,
    ),
  );
  return new Promise((resolve, reject) => {
    const [cmd, ...args] = command.split(" ");

    const spawnOptions: SpawnOptions = {
      cwd: cwd || process.cwd(), // 当前工作目录
      stdio: "inherit", // 实时输出到控制台
      shell: true, // 使用shell执行命令
    };

    let errorMessage = "";

    const child = spawn(cmd, args, spawnOptions);

    child.on("error", (err) => {
      errorMessage = err.message;
    });

    child.on("close", (code) => {
      if (code === 0) {
        console.log(chalk.bgGreen(`[exec_command 工具调用成功] 命令执行成功`));
        resolve(`
            命令执行成功:
            - 执行命令：${command}
            - 工作目录：${spawnOptions.cwd}

            注意:
            - 如何后续需要继续在当前目录下执行命令，可以继续调用exec_command工具，并且传入相同的cwd参数。
            - 不要使用cd命令
          `);
      } else {
        console.log(
          chalk.bgRed(
            `[exec_command 工具调用失败] 命令执行失败，退出码: ${code}`,
          ),
        );
        resolve(`
              命令执行失败:
            - 错误信息：${errorMessage}
            - 工作目录：${spawnOptions.cwd}
            - 退出码：${code}
            `);
      }
    });
  });
}

const execCommandToolSchema = z.object({
  command: z.string().describe("要执行的命令字符串，例如'ls -la'"),
  cwd: z
    .string()
    .optional()
    .describe("命令执行的当前工作目录，默认为process.cwd()"),
});

const execCommandTool = tool(execCommand, {
  name: "exec_command",
  description: "在指定目录下执行命令，并实时输出到控制台",
  schema: execCommandToolSchema,
});

export default execCommandTool;

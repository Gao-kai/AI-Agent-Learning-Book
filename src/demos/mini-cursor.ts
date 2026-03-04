import chalk from "chalk";
import model from "../share/model";
import execCommandTool from "../tools/exec-command";
import listDirectoryTool from "../tools/list-directory";
import readFileTool from "../tools/read-file";
import writeFileTool from "../tools/write-file";
import {
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { type DynamicStructuredTool } from "@langchain/core/tools";

const tools: DynamicStructuredTool[] = [
  readFileTool,
  writeFileTool,
  execCommandTool,
  listDirectoryTool,
];
const modelWithTools = model.bindTools(tools);

/**
 * AI Agent调用工具
 * @param userQuery
 * @param maxIterations
 */
async function runAgentWithTools(
  humanMessage: HumanMessage,
  maxIterations = 30,
) {
  const systemMessage = new SystemMessage(`
        你是一个智能、高效的AI代码助手，可以使用工具完成任务。

        当前工作目录：${process.cwd()}

        工具列表：
        - read_file：根据提供的文件路径（绝对路径或相对路径），读取文件内容
        - write_file：将内容写入指定路径的文件中，如果文件路径不存在则先创建后写入
        - list_directory：列出指定目录下的所有文件和文件夹
        - exec_command：在指定目录下执行命令，并实时输出到控制台

        注意事项：
        - 回复简洁
        - 只需要告诉我具体做了什么事情，不要输出思考过程
    `);
  const messages: BaseMessage[] = [systemMessage, humanMessage];

  for (let i = 0; i < maxIterations; i++) {
    console.log(chalk.bgBlue(`⏳AI大模型第${i + 1}轮思考中...`));
    /**
     * 1. 什么时候循环中止？
     * AI大模型返回的结果数组中tool_calls为空的时候
     */
    let response = await modelWithTools.invoke(messages);
    messages.push(response);

    if (!response.tool_calls || response?.tool_calls?.length === 0) {
      console.log(chalk.bgMagenta(`💥AI最终输出如下：\n${response.content}`));
      return response.content;
    }

    for (const tool_call of response?.tool_calls) {
      const invokeTool = tools.find((item) => item.name === tool_call.name);
      if (invokeTool) {
        const { id, name, args } = tool_call;
        const toolCallResult = await invokeTool.invoke(args);
        const toolMessage = new ToolMessage({
          content: toolCallResult,
          tool_call_id: id,
        });
        messages.push(toolMessage);
      }
    }
  }

  // 当maxIterations轮对话之后 此时输出AI大模型返回的最近一次的消息
  return messages[messages.length - 1].content;
}

const humanMessage = new HumanMessage(`
 使用Vite+React+TypeScript创建一个功能丰富的TodoList 应用：

    1. 创建项目：echo -e "n\nn" | pnpm create vite react-todo-app --template react-ts

    2. 修改 src/App.tsx，实现完整功能的 TodoList：
     - 添加、删除、编辑、标记完成
     - 分类筛选（全部/进行中/已完成）
     - 统计信息显示
     - localStorage 数据持久化

    3. 添加复杂样式：
     - 渐变背景（蓝到紫）
     - 卡片阴影、圆角
     - 悬停效果

    4. 添加动画：
     - 添加/删除时的过渡动画
     - 使用 CSS transitions

    5. 列出目录确认
    6. 安装依赖
    7. 本地启动项目
`);

runAgentWithTools(humanMessage);

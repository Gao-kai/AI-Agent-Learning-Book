import { ChatOpenAI } from "@langchain/openai";
import executeCommandTool from "./tools/execute-command-tool";
import listDirectoryTool from "./tools/list-directory-tool";
import readFileTool from "./tools/read-file-tool";
import writeFileTool from "./tools/write-file-tool";
import {
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import "dotenv/config";
import { DynamicStructuredTool } from "@langchain/core/tools";

const tools = [
  readFileTool,
  writeFileTool,
  executeCommandTool,
  listDirectoryTool,
];

const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.MODEL_NAME,
  temperature: 0,
  configuration: {
    baseURL: process.env.BASE_URL,
  },
});

const modelWithTools = model.bindTools(tools);

async function runAgentWithTools(question: string, maxIterations = 30) {
  let messages: BaseMessage[] = [];

  const systemMessage: SystemMessage = new SystemMessage(`
    你是一个AI智能助手,可以调用工具完成任务。

    当前工作目录:
    ${process.cwd()}
    
    工具列表:
    1. read-file 读取文件
    2. write-file 写入内容到指定文件
    3. exec-command 指定目录执行终端命令
    4. list-directory 列出指定目录项目名称


    回复要求:
    1. 回复要简洁，只说做了什么
`);
  const humanMessage: HumanMessage = new HumanMessage(question);
  messages.push(systemMessage, humanMessage);

  for (let i = 0; i < maxIterations; i++) {
    const response = await modelWithTools.invoke(messages);
    messages.push(response);

    if (!response.tool_calls || response.tool_calls.length === 0) {
      console.log(`最终回复：\n ${response.content}`);
      return response.content;
    }

    for (let j = 0; j < response?.tool_calls.length; j++) {
      const tool_call = response?.tool_calls[j];
      const tool: DynamicStructuredTool = tools.find(
        (item) => item.name === tool_call.name,
      );
      if (tool) {
        const toolResult = await tool.invoke(tool_call.args);
        // 工具返回
        messages.push(
          new ToolMessage({
            content: toolResult?.content,
            tool_call_id: tool_call.id,
          }),
        );
      }
    }
  }

  return messages[messages.length - 1].content;
}

const humanMessage = `
  创建一个功能丰富的 React TodoList 应用:
  
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

  注意：使用 pnpm，功能要完整，样式要美观，要有动画效果

  之后在 react-todo-app 项目中：
  1. 使用 pnpm install 安装依赖
  2. 使用 pnpm run dev 启动服务器
`;

runAgentWithTools(humanMessage);

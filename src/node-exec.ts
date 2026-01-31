import { spawn } from "node:child_process";

const command =
  "echo -e 'n\nn' | pnpm create vite react-todo-app --template react-ts";
const [cmd, ...args] = command.split(" ");
const child = spawn(cmd, args, {
  cwd: process.cwd(),
  stdio: "inherit",
  shell: true,
});

child.on("error", (error) => {
  console.error(error);
});

child.on("close", (code) => {
  if (code === 0) {
    process.exit(0);
  } else {
    process.exit(code || 1);
  }
});

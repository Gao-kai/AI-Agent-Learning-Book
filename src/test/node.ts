import fs from "node:fs/promises";
import path from "node:path";

async function test() {
  const fileNames = await fs.readdir(process.cwd());
  console.log(fileNames);
}

test();

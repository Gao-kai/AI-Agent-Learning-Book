import { MarkdownTextSplitter } from "@langchain/textsplitters";
import { getEncodingNameForModel, getEncoding } from "js-tiktoken";
import { Document } from "@langchain/core/documents";
import { readFile } from "node:fs/promises";
import { logSplitDocumentsInfo } from "./RecursiveCharacterTextSplitter";

/**
 * 获取不同AI 大模型的编码名称
 * 并且基于编码名称获取编码对象
 * 最后基于编码对象计算文本的Token数
 * 同样是两个汉字苹果和吃饭，返回的token数可能不同，因此字符个数和token个数并不是一一对应的关系
 * 这和不同模型的分词器有关
 */
const modelName = "gpt-4";
const encodingName = getEncodingNameForModel(modelName); // 'cl100k_base'
const encoding = getEncoding(encodingName); // return Encoding object

const markdown = await readFile("./records/01. AI Agent开发记录.md", "utf-8");

const markdownDocument = new Document({
  pageContent: markdown,
  metadata: {
    author: "Artest",
    date: "2026-03-08",
    source: "01. AI Agent开发记录.md",
  },
});

/**
 * MarkdownTextSplitter 基于Markdown语法规则分割文本
 * MarkdownTextSplitter是RecursiveCharacterTextSplitter的子类
 * 因此文本分割策略与RecursiveCharacterTextSplitter相同
 * 只不过分隔符内置实现了Markdown语法规则
 *
 */
const markdownTextSplitter = new MarkdownTextSplitter({
  chunkSize: 400, // 每个chunk的字符数
  chunkOverlap: 40, // 每个chunk之间的重叠字符数
});

const splitMarkdownDocuments = await markdownTextSplitter.splitDocuments([
  markdownDocument,
]);
console.log(
  `Markdown文档分割完成，共 ${splitMarkdownDocuments.length} 个分块\n`,
);

logSplitDocumentsInfo(splitMarkdownDocuments);

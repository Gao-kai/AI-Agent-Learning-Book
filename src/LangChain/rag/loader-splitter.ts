/**
 * 本文件实现将远程WEB页面内容加载为Document对象存入内存向量数据库
 * 然后在询问时基于内存向量数据库内容进行检索
 */
import { model, embeddingModel } from "../share/model";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import "cheerio";

/**
 * 创建CheerioWebBaseLoader实例
 * 指定页面URL和解析选项
 */
const cheerioLoader = new CheerioWebBaseLoader(
  "https://bbs.hupu.com/637781183.html",
  {
    selector: ".post-content_main-post-info__qCbZu .thread-content-detail p",
    timeout: 5000,
  },
);

/**
 * 加载远程WEB页面内容为Document对象
 */
const documents = await cheerioLoader.load();

/**
 * 创建RecursiveCharacterTextSplitter实例
 * 指定每个chunk的字符数、每个chunk之间的重叠字符数和分隔符
 */
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500, // 每个chunk的字符数
  chunkOverlap: 100, // 每个chunk之间的重叠字符数
  separators: ["。", ",", "\n\n", "\n", "?", "!"], // 分隔符
});

const splitDocuments = await textSplitter.splitDocuments(documents);
console.log(`文档分割完成，共 ${splitDocuments.length} 个分块\n`);
console.log("正在创建向量存储...");
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocuments,
  embeddingModel,
);
const retriever = vectorStore.asRetriever({ k: 3 });

const questions = ["勇士的未来为什么是不确定的?未来可能有何变化?"];

for (const question of questions) {
  console.log("=".repeat(80));
  console.log(`问题：${question}`);
  console.log("=".repeat(80));

  /**
   * 使用检索器从向量数据库中检索与问题相关的文档
   */
  const retrievedDocs = await retriever.invoke(question);
  console.log("使用检索器从向量数据库中检索与问题相关的文档：");

  /**
   * 使用向量数据库检索与问题相关的文档 并返回相似度分数
   */
  const scoredResults = await vectorStore.similaritySearchWithScore(
    question,
    3,
  );
  console.log("使用向量数据库检索与问题相关的文档 并返回相似度分数：");

  retrievedDocs.forEach((doc, index) => {
    const scoredResult = scoredResults.find(([document, score]) => {
      return document.pageContent === doc.pageContent;
    });

    if (scoredResult) {
      const score = scoredResult[1];
      const similarity = score !== null ? (1 - score).toFixed(4) : "N/A";

      console.log(`文档${index + 1} 相似度分数：${similarity}`);
      console.log(`文档${index + 1} 内容：${doc.pageContent}`);
      console.log(`文档${index + 1} 元数据：${JSON.stringify(doc.metadata)}`);
    }
  });

  console.log("=".repeat(80));

  const context = retrievedDocs
    .map((doc, index) => {
      return `文档${index + 1}：${doc.pageContent}`;
    })
    .join("\n\n-----\n\n");

  const prompt = `
    你是一个擅长体育NBA的文章阅读助手，请根据文章内容进行作答。
    如果文章中没有提到，就说"文章中没有提到这个细节"。

  问题：${question}
  相关文档：${context}
  `;

  console.log("\n【AI 回答】");
  const response = await model.invoke(prompt);
  console.log(response.content);
  console.log("\n");
}

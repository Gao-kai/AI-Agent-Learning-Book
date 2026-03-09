/**
 * 本文件实现从虎扑湖人专区24小时榜加载新闻和评论到向量数据库
 */
import { model, embeddingModel } from "../share/model";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import * as cheerio from "cheerio";

/**
 * 从虎扑湖人专区获取24小时榜的新闻链接
 */
async function getLakersHotPosts() {
  const loader = new CheerioWebBaseLoader("https://bbs.hupu.com/lakers-hot");
  const documents = await loader.load();
  
  const html = documents[0].pageContent;
  
  // 先打印HTML的前1000个字符，了解页面结构
  console.log("页面HTML结构预览:");
  console.log(html.substring(0, 1000) + "...");
  
  const $ = cheerio.load(html);
  
  // 查找24小时榜的帖子链接
  const posts = [];
  
  // 分析页面结构后，找到正确的选择器
  // 尝试查找所有可能的帖子链接
  $(".title a").each((index, element) => {
    if (index < 50) { // 只获取前50条
      const title = $(element).text().trim();
      const url = $(element).attr("href");
      
      if (title && url) {
        // 确保URL是完整的
        const fullUrl = url.startsWith("http") ? url : `https://bbs.hupu.com${url}`;
        posts.push({ title, url: fullUrl });
      }
    }
  });
  
  return posts;
}

/**
 * 加载单个帖子的内容和评论
 */
async function loadPostContent(url: string) {
  try {
    const loader = new CheerioWebBaseLoader(url, {
      timeout: 10000,
    });
    
    const documents = await loader.load();
    const html = documents[0].pageContent;
    const $ = cheerio.load(html);
    
    // 提取帖子标题
    const title = $("h1").text().trim();
    
    // 提取帖子内容
    const content = $("#main-content").text().trim();
    
    // 提取评论
    const comments = [];
    $("#comment-list").find(".comment-item").each((index, element) => {
      const commentContent = $(element).find(".comment-content").text().trim();
      if (commentContent) {
        comments.push(commentContent);
      }
    });
    
    // 构建完整内容
    let fullContent = `标题: ${title}\n\n内容: ${content}\n\n评论:\n${comments.join("\n")}`;
    
    return new Document({
      pageContent: fullContent,
      metadata: {
        source: url,
        title: title,
        type: "lakers-news",
      },
    });
  } catch (error) {
    console.error(`加载帖子失败: ${url}`, error);
    return null;
  }
}

/**
 * 主函数：加载虎扑湖人专区24小时榜的新闻和评论到向量数据库
 */
async function loadLakersNewsToVectorStore() {
  console.log("正在获取虎扑湖人专区24小时榜新闻...");
  const posts = await getLakersHotPosts();
  console.log(`获取到 ${posts.length} 条新闻`);
  
  console.log("正在加载新闻内容和评论...");
  const documents: Document[] = [];
  
  for (const post of posts) {
    console.log(`正在加载: ${post.title}`);
    const doc = await loadPostContent(post.url);
    if (doc) {
      documents.push(doc);
    }
  }
  
  console.log(`成功加载 ${documents.length} 条新闻`);
  
  if (documents.length === 0) {
    console.log("没有加载到任何新闻，程序退出");
    return;
  }
  
  console.log("正在分割文档...");
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
    separators: ["。", ",", "\n\n", "\n", "?", "!"],
  });
  
  const splitDocuments = await textSplitter.splitDocuments(documents);
  console.log(`文档分割完成，共 ${splitDocuments.length} 个分块`);
  
  console.log("正在创建向量存储...");
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocuments,
    embeddingModel
  );
  
  console.log("向量存储创建完成");
  
  // 测试查询
  const retriever = vectorStore.asRetriever({ k: 3 });
  const testQuestions = ["湖人最近的比赛情况如何？", "詹姆斯的表现怎么样？"];
  
  for (const question of testQuestions) {
    console.log("\n" + "=".repeat(80));
    console.log(`测试问题：${question}`);
    console.log("=".repeat(80));
    
    const retrievedDocs = await retriever.invoke(question);
    
    console.log("检索到的相关文档：");
    retrievedDocs.forEach((doc, index) => {
      console.log(`\n文档${index + 1}:`);
      console.log(`标题: ${doc.metadata.title}`);
      console.log(`内容: ${doc.pageContent.substring(0, 200)}...`);
    });
  }
}

// 执行主函数
loadLakersNewsToVectorStore().catch(console.error);

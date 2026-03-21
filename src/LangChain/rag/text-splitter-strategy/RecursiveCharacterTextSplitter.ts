/**
 * 文本分割策略
 */
import {
  RecursiveCharacterTextSplitter,
  MarkdownTextSplitter,
  LatexTextSplitter,
} from "@langchain/textsplitters";
import { getEncodingNameForModel, getEncoding } from "js-tiktoken";
import { Document } from "@langchain/core/documents";

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
const logDocuments = new Document({
  pageContent: `
    [2024-01-15 10:00:00] INFO: Application started
    [2024-01-15 10:00:05] DEBUG: Loading configuration file
    [2024-01-15 10:00:10] INFO: Database connection established
    [2024-01-15 10:00:15] WARNING: Rate limit approaching
    [2024-01-15 10:00:20] ERROR: Failed to process request
    [2024-01-15 10:00:25] INFO: Retrying operation
    [2024-01-15 10:00:30] SUCCESS: Operation completed
    [2026-01-10 14:30:00] INFO: 系统开始执行大规模数据迁移任务，本次迁移涉及核心业务数据库中的用户表、订单表、商品库存表、物流信息表、支付记录表、评论数据表等共计十二个关键业务表，预计处理数据量约500万条记录，数据总大小预估为280GB，迁移过程将采用分批次增量更新策略以减少对生产环境的影响，同时启用双写机制确保数据一致性，任务预计总耗时约3小时15分钟，迁移完成后将自动触发全面的数据一致性校验流程以及性能基准测试，请相关运维人员和DBA团队密切关注系统资源使用情况、网络带宽占用率以及任务执行进度，如遇异常情况请立即启动应急预案并通知技术负责人

`,
  metadata: {
    source: "日志文件",
    date: "2024-01-15",
  },
});

/**
 * RecursiveCharacterTextSplitter优先使用\n 换行符分割
 * 当分割后的chunk大小还是大于chunkSize时，再使用其他分隔符递归分割
 * chunksize默认指的是字符数，而不是Token数
 * 但是可以通过指定lengthFunction函数来设置成为基于Token数来分割
 */
const logTextSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200, // 每个chunk的字符数
  chunkOverlap: 20, // 每个chunk之间的重叠字符数
  separators: ["\n", ".", ",", "，"], // 设置分隔符,
  lengthFunction: (text) => encoding.encode(text).length, // 设置基于Token数来分割
});

const splitDocuments = await logTextSplitter.splitDocuments([logDocuments]);
console.log(`日志文档分割完成，共 ${splitDocuments.length} 个分块\n`);

// logSplitDocumentsInfo(splitDocuments);

/**
 * 打印分割后的文档数组的信息
 * @param splitDocuments 分割后的文档数组
 */
export function logSplitDocumentsInfo(splitDocuments: Document[]) {
  for (const document of splitDocuments) {
    console.log("=".repeat(80));
    console.log(`分块内容：${document.pageContent}`);
    console.log(`分块字符数：${document.pageContent.length}`);
    console.log(`分块Token数：${encoding.encode(document.pageContent).length}`);
    console.log("=".repeat(80));
  }
}

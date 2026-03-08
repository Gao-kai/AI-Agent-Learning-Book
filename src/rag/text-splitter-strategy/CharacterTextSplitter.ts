/**
 * 文本分割策略
 */
import { CharacterTextSplitter } from "@langchain/textsplitters";
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
console.log("apple", encoding.encode("apple").length);
console.log("pineapple", encoding.encode("pineapple").length);
console.log("苹果", encoding.encode("苹果").length);
console.log("吃饭", encoding.encode("吃饭").length);
console.log("一二三", encoding.encode("一二三").length);

/**
 * RecursiveCharacterTextSplitter 递归字符文本分割器
 * 递归地将文本分割为多个块，每个块的字符数不超过指定的 chunkSize。
 * 如果文本超过 chunkSize，会递归地将文本按照指定的分隔符分割为多个块，每个块的字符数不超过 chunkSize
 *
 * CharacterTextSplitter 字符文本分割器
 * 严格按照指定的分隔符将文本分割为多个块，每个块的字符数不超过 chunkSize
 * 不会递归地分割文本，只是简单地按照分隔符分割
 *
 * TokenTextSplitter 令牌文本分割器
 * 按照指定的令牌数将文本分割为多个块，每个块的令牌数不超过 chunkSize
 * 令牌数是根据模型的令牌 计算得到的，不同的模型有不同的令牌
 * 例如，OpenAI 的模型使用的是 OpenAI 的令牌，而 Google 的模型使用的是 Google 的令牌izer
 */

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

const logTextSplitter = new CharacterTextSplitter({
  chunkSize: 200, // 每个chunk的字符数
  chunkOverlap: 20, // 每个chunk之间的重叠字符数
  separator: "\n", // 分隔符
});

const splitDocuments = await logTextSplitter.splitDocuments([logDocuments]);
console.log(`日志文档分割完成，共 ${splitDocuments.length} 个分块\n`);

/**
 * text-splitter的原则是优先保证上下文的语义完整性
 * 因此就算某一个chunk的size没有达到chunkSize，宁愿chunk小一点浪费空间也会优先保证上下文的语义完整性
 *
 * CharacterTextSplitter 非常死板，你告诉它按照换行符分割，它就会严格按照这个，就算超过了 chunk size 也不拆分。
 */
for (const document of splitDocuments) {
  console.log("=".repeat(80));
  console.log(`分块内容：${document.pageContent}`);
  console.log(`分块字符数：${document.pageContent.length}`);
  console.log(`分块Token数：${encoding.encode(document.pageContent).length}`);
  console.log("=".repeat(80));
}

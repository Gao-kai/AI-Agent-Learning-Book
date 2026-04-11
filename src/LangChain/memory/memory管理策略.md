# Memory 的管理策略

1. 可以手动构建Message数组，每次和AI大模型交互时，手动添加Message到数组中。但是这种方法缺点很明显：

- 需要手动管理Message数组
- Message数组中存储的上下文长度可能超出模型上限

有同学说，可以只保留最近的几条 message，之前的舍弃掉啊。
有同学说，直接舍弃之前的也不好，可以对之前的做一些总结，保留这个总结和最近的几条 message。
有同学说可以用我们刚学的向量数据库啊，根据语义检索之前的 message

没错，主流的也就是这三种思路，截断、总结、检索
达到限制自动触发总结

存储层，
逻辑层，也就是截断、总结、向量数据库这些：

每个 xxMemory 类都有一个 chatHistory 属性，关联着存储层。
 @langchain/classic 废弃

 截断就是根据总 token 数量来保留最近的 message
 总结就是调用大模型对之前的 message 生成一个摘要
 检索向量数据库就是之前的 RAG 流程，只不过用来对 message 做语义检索

  trimMessages 的 api，可以根据 token 来截断消息
  history + trimMessages 的 api

js-tiktoken 核心 API总结

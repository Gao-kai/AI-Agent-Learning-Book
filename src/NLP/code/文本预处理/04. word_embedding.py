"""
文本向量化之 word embedding

Word2Vec的原理是：对输入的文本进行训练，先将文本转化为词向量表示，然后再带入模型做其他操作，词向量是静态的
Word Embedding的原理是：对输入的文本进行训练，训练的过程中词向量会基于词嵌入层的作用自动生成。
"""
import torch # 深度学习框架 用于构建神经网络模型 封装了张量操作
import jieba # 中文分词工具
import torch.nn as nn # 神经网络模块
from tensorflow.keras.preprocessing.text import Tokenizer # 词汇映射器
from torch.utils.tensorboard import SummaryWriter # 用于可视化模型训练过程中的指标

"""
使用 tensorboard 可视化 词向量
"""
def word_embedding_show():

    # 获取原始文本
    text = "今天我真好看，我真的真的想和你一起去爬山"

    # 对原始文本进行分词 切分成一个个的Token
    words = jieba.lcut(text, cut_all=False)
    print(f"words: {words}")

    # 创建词频统计表 将words中的单词去重-计算词频-构建词汇表
    # 支持 索引-单词 映射 以及 单词-索引 映射
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)
    print(f"tokenizer.word_index: {tokenizer.word_index}")
    print(f"tokenizer.index_word: {tokenizer.index_word}")

    # 获取词汇表中每一个单词对应的索引列表
    tokens = tokenizer.word_index.values()
    print(f"tokens: {tokens}")

    # 将原始文本转化为索引表示
    # 假设原始文本为“你今天真好看” => 转化为索引 [1, 2, 3, 4, 5]
    seq2id = tokenizer.texts_to_sequences([words])
    print(f"seq2id: {seq2id}")

    # 创建词嵌入层 本质是创建一个随机的词向量矩阵 后续会不断通过反向传播来优化这个矩阵
    # num_embeddings: 词汇表大小 也就是去重后词的个数
    # embedding_dim: 词向量维度
    # 词嵌入层的矩阵维度为“词汇表大小 * 词向量维度”
    # 例如：词汇表大小为10000，词向量维度为8，那么词嵌入层的矩阵维度为“10000 * 8”
    embedding_layer = nn.Embedding(num_embeddings=len(tokens), embedding_dim=8)

    # 查看词嵌入层的权重参数 也就是词向量矩阵
    print(f"embedding_layer.weight.data: {embedding_layer.weight.data}") 
    print(f"embedding_layer.weight.shape: {embedding_layer.weight.shape}") 

    # 按照词频依次查看每个单词的词向量表示
    for index in range(len(tokenizer.word_index)):
      # 获取当前单词的词向量
      word_vector = embedding_layer(torch.tensor(index))

      # 获取当前索引对应的单词
      word = tokenizer.index_word[index+1]
      print(f"单词为: {word} 词向量表示为: {word_vector}")

if __name__ == "__main__":
    word_embedding_show()
"""
将文本转化为词向量方式之Word2Vec

Word2Vec主要有两种模型
1. CBOW 连续词袋模型 基于上下文预测中间词
2. Skip-gram 跳词模型 基于中间词预测上下文

无论使用哪种模型，都是使用深度学习中的隐藏层的权重矩阵来充当词向量矩阵。也就是：
权重矩阵的每一列，分别对应1个单词的word2vec词向量
facebook的Fasttext库是一个开源的词向量和文本分类库,我们可以基于此工具来演示词向量模型。
"""
import fasttext
import os

"""
训练词向量模型并保存模型
"""
def train_word2vec_save():
    # 获取原数据绝对路径
    abs_data_path = os.path.join(os.path.dirname(__file__), '../../data/wh02ad')

    # 训练模型（默认参数model为skipgram）
    model = fasttext.train_unsupervised(abs_data_path)

    # 保存模型
    abs_model_path = os.path.join(os.path.dirname(__file__), '../../models/wh02_word2vec.model')
    model.save_model(abs_model_path)


"""
加载词向量模型并查看单个词的词向量表示
"""
def load_word2vec_model():
    # 加载预训练的fasttext模型
    abs_model_path = os.path.join(os.path.dirname(__file__), '../../models/wh02_word2vec.model')
    model = fasttext.load_model(abs_model_path)

    # 获取单个词的词向量表示
    result = model.get_word_vector("manufacturing")

    # 打印信息
    print(f'查询出来的词向量表示为：{result}')
    print(f'查询出来的词向量类型为：{type(result)}') # <class 'numpy.ndarray'>
    print(f'查询出来的词向量形状为：{result.shape}') # (100,)


"""
测试单词的相似度，评估模型效果
"""
def get_word_similarity():
    # 加载预训练的fasttext模型
    abs_model_path = os.path.join(os.path.dirname(__file__), '../../models/wh02_word2vec.model')
    model = fasttext.load_model(abs_model_path)

    # 获取某个单词的最相似的近义词
    # 默认是10个 可以用于检测模型的语义理解能力
    # 返回格式为元祖列表：[(相似度分数，近义词)，（相似度分数，近义词）, ...]
    result = model.get_nearest_neighbors("county")
    print(f'查询出来的最相似的近义词为：{result}')

"""
模型超参数设定train_unsupervised
1. input: 训练数据路径 output: 训练好的模型对象 model
2. model 词向量模型类型，默认skipgram，可以设置为cbow或skipgram
3. dim 词向量维度，默认100
4. epoch 训练轮数，默认5
5. lr 学习率，默认0.01
6. thread 线程数，默认是最大CPU线程数-1
"""
def set_hyper_params():
    # 获取原数据绝对路径
    abs_data_path = os.path.join(os.path.dirname(__file__), '../../data/fil9')

    model = fasttext.train_unsupervised(
        input = abs_data_path,
        model = 'cbow',
        dim = 100,
        epoch = 1,
        lr = 0.01,
        thread = 8,
    )

    # 保存模型
    abs_model_path = os.path.join(os.path.dirname(__file__), '../../models/file09_word2vec_cbow.model')
    model.save_model(abs_model_path)

if __name__ == '__main__':
    # 1. 训练词向量模型
    # train_word2vec_save()

    # 2. 加载词向量模型并查看词向量表示
    # load_word2vec_model()

    # 3. 测试单词的相似度
    # get_word_similarity()

    # 4. 模型超参数设定
    set_hyper_params()

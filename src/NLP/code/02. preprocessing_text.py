"""
词张量化表示方法
"""
import os
# 导入TensorFlow的Keras库的Tokenizer类
from tensorflow.keras.preprocessing.text import Tokenizer
# 导入joblib库
import joblib


def create_one_hot_encode():
    # 创建词表
    vocabulary = {"你","今天","真","好看"}

    # 创建Tokenizer实例化对象
    tokenizer = Tokenizer()

    # 对文本序列进行训练
    tokenizer.fit_on_texts(vocabulary)

    # 打印词表
    print(tokenizer.word_index)
    print(tokenizer.index_word)

    # 查找每个分词的one-hot编码
    for word in vocabulary:
      # 初始化一个全0的列表 长度为词表的大小
      zero_list = [0] * len(vocabulary)
      # 查找当前分词的索引
      index = tokenizer.word_index[word]
      # 将索引位置的元素设为1
      zero_list[index-1] = 1 
      print(f"{word} 的one-hot编码为：{zero_list}")
    
    # 将训练好的Tokenizer对象保存到文件 避免每次训练都重新创建
    current_file_path = os.path.abspath(__file__)
    current_file_dirname = os.path.dirname(current_file_path)
    save_path = os.path.join(current_file_dirname,'../libs/tokenizer')
    joblib.dump(tokenizer, save_path)


def get_token_one_hot_encode(token):
    vocabulary = {"你","今天","真","好看"}

    # 从文件中加载训练好的Tokenizer对象
    save_path = os.path.join(os.path.dirname(__file__), '../libs/tokenizer')
    tokenizer = joblib.load(save_path)

    # 查找当前分词的索引
    index = tokenizer.word_index[token]
    # 初始化一个全0的列表 长度为词表的大小
    zero_list = [0] * len(vocabulary)
    # 将索引位置的元素设为1
    zero_list[index-1] = 1 
    print(f"{token} 的one-hot编码为：{zero_list}")




if __name__ == '__main__':
    # 创建one-hot编码并保存训练结果
    # create_one_hot_encode()

    # 获取某个分词的one-hot编码
    get_token_one_hot_encode("你")

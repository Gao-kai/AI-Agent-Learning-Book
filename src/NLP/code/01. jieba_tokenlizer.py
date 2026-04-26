import jieba
import jieba.posseg as pseg
import os

"""
中文分词工具jieba使用
1. 精确模式
2. 全模式
3. 搜索引擎模式
"""


def precise_mode(text):
    """
    精确模式 - 最常用的分词模式
    
    特点：
    - 试图将句子最精确地切分
    - 适合文本分析、关键词提取等场景
    - 不存在冗余词汇
    
    :param text: 待分词的中文文本字符串
    :return: 分词结果列表
    """
    # jieba.lcut() 返回分词结果列表，cut_all=False 表示精确模式（默认）
    result = jieba.lcut(text, cut_all=False)
    return result


def full_mode(text):
    """
    全模式 - 穷举所有可能的分词组合
    
    特点：
    - 扫描所有可能的分词结果
    - 速度快，但存在大量冗余词汇
    - 适合粗粒度的分词场景，如关键词检索
    
    :param text: 待分词的中文文本字符串
    :return: 分词结果列表（包含所有可能的分词）
    """
    # cut_all=True 表示全模式
    result = jieba.lcut(text, cut_all=True)
    return result


def search_engine_mode(text):
    """
    搜索引擎模式 - 专为搜索引擎设计的分词模式
    
    特点：
    - 在精确模式基础上，对长词进行再次切分
    - 提高召回率，适合搜索引擎构建倒排索引
    - 兼顾精确性和召回率
    
    :param text: 待分词的中文文本字符串
    :return: 分词结果列表
    """
    # jieba.lcut_for_search() 专门用于搜索引擎分词
    result = jieba.lcut_for_search(text)
    return result


def cut_method_iterator(text):
    """
    cut方法 - 返回迭代器对象（区别于lcut返回列表）
    
    关键区别：
    - jieba.cut() 返回迭代器，内存效率更高
    - jieba.lcut() 返回列表，方便直接使用但占用更多内存
    
    :param text: 待分词的中文文本字符串
    :return: 分词结果迭代器对象
    """
    # jieba.cut() 返回迭代器，cut_all=False 表示精确模式
    result = jieba.cut(text, cut_all=False)
    return result


def consume_iterator(iterator):
    """
    演示Python迭代器的多种消费方式
    
    迭代器特点：
    - 惰性计算：只在需要时生成下一个元素
    - 一次性消费：遍历后迭代器为空，不能重复使用
    - 内存高效：不需要一次性加载所有元素到内存
    
    :param iterator: 待消费的迭代器对象
    :return: None
    """
    print("【迭代器消费方式演示】")
    print()
    
    # 方式1：for循环遍历（最常用）
    print("1. for循环遍历：")
    for word in iterator:
        print(f"   {word}", end=" ")
    print()
    print()
    
    # 注意：迭代器已被消费完毕，需要重新创建
    new_iterator = jieba.cut("我爱自然语言处理", cut_all=False)
    
    # 方式2：强制转换为列表
    print("2. list()转换为列表：")
    word_list = list(new_iterator)
    print(f"   结果：{word_list}")
    print()
    
    # 方式3：next()函数逐个获取
    print("3. next()函数逐个获取：")
    it = jieba.cut("人工智能", cut_all=False)
    try:
        while True:
            word = next(it)
            print(f"   获取到：{word}")
    except StopIteration:
        print("   迭代器已耗尽")
    print()
    
    # 方式4：enumerate()带索引遍历
    print("4. enumerate()带索引遍历：")
    it = jieba.cut("机器学习", cut_all=False)
    for index, word in enumerate(it):
        print(f"   索引{index}：{word}")
    print()
    
    # 方式5：*解包操作
    print("5. *解包操作：")
    it = jieba.cut("深度学习", cut_all=False)
    print(f"   解包结果：{[*it]}")


def custom_dict(text):
    """
    自定义词典 - 加载自定义词典文件
    
    说明：
    - 自定义词典文件格式：每个词占一行，每个词之间用空格分隔 依次表示词(必填)、词频(可选)、词性(可选)
    - 例如：云计算 100 n 
    - 使用自定义字典后，会优先使用自定义词典中的词进行分词，如果自定义词典中没有，会使用默认词典中的词进行分词

    注意：
    - 自定义词典文件编码必须为UTF-8
    - jieba会加载默认的词典库，如果自定义加载的词典中词语和默认库冲突，会按照词频最大的那个词进行分词
    
    :return: None
    """
    # 加载自定义词典（使用绝对路径确保可靠性）
    dict_path = os.path.join(os.path.dirname(__file__), "../dict/user-dict.txt")
    jieba.load_userdict(dict_path)

    # 基于自定义词典进行分词
    result_by_custom_dict = jieba.lcut(text, cut_all=False)

    return result_by_custom_dict


def pos_tagging(text):
    """
    词性标注 - 为分词后的每个词标注词性

    特点：
    - 基于HMM模型和Viterbi算法
    - 同时返回词语和对应的词性标签
    - 适合词性统计、句法分析等场景

    常用词性标签：
    - n: 名词
    - nr: 人名
    - ns: 地名
    - nt: 时间机构名
    - nz: 其他专名
    - v: 动词
    - vd: 副动词
    - vn: 名动词
    - a: 形容词
    - ad: 副形词
    - an: 名形词
    - d: 副词
    - m: 量词
    - q: 量词
    - r: 代词
    - p: 介词
    - c: 连词
    - u: 助词
    - xc: 其他虚词
    - w: 标点符号
    -PER: 人名
    -LOC: 地名
    -TIME: 时间

    :param text: 待标注的中文文本字符串
    :return: 分词及词性标注结果列表，每个元素为(word, flag)元组
    """
    result = pseg.lcut(text)
    return result


if __name__ == "__main__":
    # 测试文本
    test_text = "我爱自然语言处理和人工智能"
    
    # 演示三种分词模式
    print("【测试文本】", test_text)
    print() 
    
    
    print("1. 精确模式：")
    precise_result = precise_mode(test_text)
    print("   结果：", precise_result)
    print() 
    # 结果： ['我', '爱', '自然语言', '处理', '和', '人工智能']
    
    print("2. 全模式：")
    full_result = full_mode(test_text)
    print("   结果：", full_result)
    print() 
    # 结果： ['我', '爱', '自然', '自然语言', '语言', '处理', '和', '人工', '人工智能', '智能']
    
    print("3. 搜索引擎模式：")
    search_result = search_engine_mode(test_text)
    print("   结果：", search_result) 
    print()
    
    # 演示cut方法（返回迭代器）
    print("4. cut方法（返回迭代器）：")
    iterator_result = cut_method_iterator(test_text)
    print("   迭代器类型：", type(iterator_result))
    print("   迭代器内容：", list(iterator_result))
    print()
    
    # 演示迭代器消费方式
    consume_iterator(jieba.cut("迭代器消费演示", cut_all=False))

    # 演示自定义词典
    result_by_custom_dict = precise_mode('当前时代，云计算和云原生成为了AI大模型的主流应用。')
    print("   结果：", result_by_custom_dict)
    print()

    # 演示词性标注
    print("5. 词性标注：")
    pos_result = pos_tagging(test_text)
    print("   结果：", pos_result)
    print("   格式化输出：")
    for word, flag in pos_result:
        print(f"   {word}/{flag}", end=" ")
    print()
    
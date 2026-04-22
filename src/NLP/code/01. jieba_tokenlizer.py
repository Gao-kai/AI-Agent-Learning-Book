import jieba

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
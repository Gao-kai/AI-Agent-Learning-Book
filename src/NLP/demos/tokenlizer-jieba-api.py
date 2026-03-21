import warnings

# 在导入jieba之前设置警告过滤器，以消除pkg_resources弃用警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

import jieba

text = "小米毕业于北京大学计算机系"

# 精确模式 对输入文本进行分词，返回一个包含精确切分后的列表
# @param {string} text - 输入的待分词文本
# @param {boolean} cut_all - 是否使用全模式分词，默认为false

#  使用cut直接返回列表，无需再转换
seg_list = jieba.lcut(text)

print("分词结果(空格分隔):", " ".join(seg_list))

# 使用lcut返回可迭代的生成器
seg_list_generator = jieba.cut(text)
for seg in seg_list_generator:
    print("分词结果:", seg)



# 全模式 对输入文本进行分词，返回一个包含所有分词结果的列表
# @param {string} text - 输入的待分词文本
# @param {boolean} cut_all - 是否使用全模式分词，默认为false
# @returns {Array<string>} - 包含所有分词结果的列表

seg_list = jieba.lcut(text, cut_all=True)
print("全模式分词结果:", seg_list)

for seg in jieba.cut(text, cut_all=True):
    print("全模式分词结果:", seg)

# 搜索引擎模式 对输入文本进行分词，返回一个包含所有分词结果的列表
# @param {string} text - 输入的待分词文本
# @param {boolean} cut_all - 是否使用全模式分词，默认为false
# @returns {Array<string>} - 包含所有分词结果的列表

seg_list = jieba.lcut_for_search(text)
print("搜索引擎模式分词结果:", seg_list)

for seg in jieba.cut_for_search(text):
    print("搜索引擎模式分词结果:", seg)


customText = "随着云原生技术的发展和普及，越来越多的企业开始采用云原生架构来部署服务，并借助大模型能力提升智能化水平，实现业务流程的自动化与智能决策。"
customSegList = jieba.lcut(customText);
print("自定义文本分词结果:", customSegList)
# 自定义文本分词结果: 
# ['随着', '云', '原生', '技术', '的', '发展', '和', '普及', '，', '越来越', '多', '的', '企业', '开始', '采用', '云', '原生', '架构', '来', '部署', '服务', '，', '并', '借助', '大', '模型', '能力', '提升', '智能化', '水平', '，', '实现', '业务流程', '的', '自动化', '与', '智能', '决策', '。']

# 加载自定义词典
jieba.load_userdict("./src/demos/user-dict.txt")

# 加载自定义词典后，再次对自定义文本进行分词
customSegList = '\n'.join(jieba.lcut(customText));
print("自定义文本分词结果:", customSegList)
# 加载自定义词典后自定义文本分词结果: 
# ['随着', '云原生', '技术', '的', '发展', '和', '普及', '，', '越来越', '多', '的', '企业', '开始', '采用', '云原生', '架构', '来', '部署', '服务', '，', '并', '借助', '大模型', '能力', '提升', '智能化', '水平', '，', '实现', '业务流程', '的', '自动化', '与', '智能', '决策', '。']

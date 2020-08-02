import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import jieba.posseg;
import jieba.analyse;
import codecs,sys;  #codecs是python的编码器基类
from sklearn import feature_extraction;
from sklearn.feature_extraction.text import TfidfTransformer;
from sklearn.feature_extraction.text import CountVectorizer;

#数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text , stopkey):
    l = [];
    pos = ['n' , 'nz' , 'v' , 'vd' , 'vn' , 'l' , 'a' , 'd']   #定义选取的词性,a形容词，d副词，l习用词，vd副动词，vn名动词
    seg = jieba.posseg.cut(text)      #分词
    for i in seg:
        if i.word not in stopkey and i.flag in pos:         #去停用词+词性筛选
            l.append(i.word);

    return l;

#tf_idf获取文本top10关键词
def getKeywords_tfidf(data , stopkey , topK):
    idList , titleList , abstractList = data['id'],data['title'],data['abstract'];  #根据下标，标题，摘要读出内容
                                                                                    #len(idList) = 10
    corpus = []    #结果
    for index in range(len(idList)):
        text = '%s。%s' % (titleList[index] , abstractList[index])   #拼接标题和摘要  永磁电机驱动的纯电动大巴车坡道起步防溜策略。(。之前是标题)本发明公开了一种永磁电机驱动的纯电动大巴车坡道起步防溜策略，
                                                                                   # 即本策略当制动踏板已踩下、永磁电机转速小于设定值并持续一定时间，整车控制单元产生一个刹车触发信号

        text = dataPrepos(text , stopkey)      #文本预处理， 将每一个结合成的文本，进行文本的预处理，并避过停用词
                                                #['永磁', '电机', '驱动', '纯', '电动', '大巴车', '坡道', '起步', '防溜', '策略', '本发明', '永磁', '电机', '驱动', '纯'.....

        text = " ".join(text)      #链接成字符串，以空格为隔断
                                    #永磁 电机 驱动 纯 电动 大巴车 坡道 起步 防溜 策略 本发明 永磁 电机 驱动 纯 电动 大巴.....

        corpus.append(text);
    # corpus的内容 ['永磁 电机 驱动 纯 电动 大巴车 坡道 起步 防溜 策略 本发明 永磁...','.....','....'.....]，最后变为1*10的矩阵

    #1，构建词频矩阵，将文本中的词语转化为词频矩阵
    vectorizer = CountVectorizer();
    X = vectorizer.fit_transform(corpus);       #词频矩阵，a[i][j]:表示j词在第i个文本中的词频
                                                #(0, 187)	1    只展示少部分，0代表第0列，187代表索引，(将所有字符做成一个索引号)是字符{"永磁":1,"电机":2, ....}的索引号，使用X.vocabulary_ 可以查看索引号
                                                #(0, 62)	1    后面的1代表频次，出现的次数，包含237个索引项
                                                #(0, 84)	1
                                                #(0, 42)	3
                                                #(0, 204)	2
    #他会转化为类似的稀疏矩阵
    #['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    #[[0 1 1 1 0 0 1 0 1]
    # [0 1 0 1 0 2 1 0 2]
    #[0 1 1 1 0 0 1 0 1]]

    #2，统计每个词的tf_idf的权值
    transformer = TfidfTransformer();
    tfidf = transformer.fit_transform(X);  #计算权值，tf计算该词在本文本中的出现频率，idf逆文件频率，log(总文本 / (1+出现的文本数量))

    #3，获取词袋模型中的关键词
    word = vectorizer.get_feature_names();     #分有237个词 ['一定', '一段时间', '主体', '乘客', '乘车', '事件', '二者', '互相', '交叉'....]

    #4，获取tf_idf矩阵，a[i][j]:表示j词在第i个文本中的tf_idf权重
    weight = tfidf.toarray();   #生成10 * 237 的权值矩阵
                                #[[0.05483076 0.         0.         ... 0.0815585  0.         0.05483076]
                                #[0.         0.         0.         ... 0.         0.         0.        ]
                                # [0.         0.         0.         ... 0.         0.         0.        ]
                                # ...
                                # [0.         0.         0.         ... 0.04569533 0.         0.        ]
                                # [0.         0.         0.         ... 0.06393673 0.         0.        ]
                                # [0.         0.         0.         ... 0.         0.         0.        ]]


    #5，打印词语权重
    ids , titles , keys = [],[],[];
    for i in range(len(weight)):
        print("------------这里输出第",i+1,"篇文本的词语tf_idf---------------")
        ids.append(idList[i]);
        titles.append(titleList[i]);
        df_word,df_weight = [],[];     #当前文章的所有词汇列表，词汇对应权重表
        for j in range(len(word)):     #len(word) = 237
            print(word[j] , weight[i][j]);
            df_word.append(word[j]);
            df_weight.append(weight[i][j]);
        df_word = pd.DataFrame(df_word,columns=['word']);
        df_weight = pd.DataFrame(df_weight,columns=['weight']);
        word_weight = pd.concat([df_word,df_weight] ,axis = 1)                 #拼接词汇表和权重列表
        word_weight = word_weight.sort_values(by = "weight" ,ascending=False)  #按照权重降序排序
        keyword = np.array(word_weight['weight']);                             #按照词汇列转成数组格式
        word_split = [keyword[x] for x in range(0,topK)]                       #抽取钱topk个词汇作为关键词
        word_split = " ".join('%s' %id for id in word_split)                   #拼接在一起,list包含数字，不能直接转化成字符串,遍历list的元素，把他转化成字符串。这样就能成功输出。
        keys.append(word_split.encode("utf-8"));

    result = pd.DataFrame({"id":ids , "title" : titles , "key" : keys} , columns=[ 'id' , 'title' , 'key'])
    return result;

def main():
    #读取数据集
    dataFile = 'sample_data.csv';
    data = pd.read_csv(dataFile);

    #停用词表
    stopkey = [w.strip() for w in codecs.open('stopWord.txt','rb').readlines()];   #停用词是指在信息检索中，为节省存储空间和提高搜索效率，在处理自然语言数据（或文本）之前或之后会自动过滤掉某些字或词，
                                                                                    # 这些字或词即被称为Stop Words（停用词）。这些停用词都是人工输入、非自动化生成的，生成后的停用词会形成一个停用词表
                                                                                    #读出所有的停用词，并使用codecs进行编码解读，防止乱码

    #tf-idef关键词抽取
    result = getKeywords_tfidf(data , stopkey , 10);
    result.to_csv("keys_tfidf.csv" , index = False);

if __name__ == '__main__':
    main();
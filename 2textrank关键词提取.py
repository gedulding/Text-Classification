import sys
import pandas as pd;
import numpy as np;
import jieba.analyse;

#计算权重进行关键词提取

def getKeywords_textrank(data , topK):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']  #提取相应的标签数据
    ids, titles, keys = [], [], [];     #结果存储
    for index in range(len(idList)):
        text = '%s。%s' % (titleList[index] , abstractList[index])    #按照题目和摘要拼接句子，句号之前是题目
        jieba.analyse.set_stop_words("stopWord.txt");      #加载自定义的停用词表
        print("\"",titleList[index],"\"" , "10 Keywords - TextRank : ");
        keywords = jieba.analyse.textrank(text , topK = topK , allowPOS=('n','nz','v','vd','vn','l','a','d'));   #关键词提取
        print("********************* keywords **********************");
        print(keywords);
        print("********************* ******** **********************");
        word_spilt = " ".join('%s' %id for id in keywords)
        keys.append(word_spilt.encode("utf-8"));
        ids.append(idList[index]);
        titles.append(titleList[index]);

    result = pd.DataFrame({"id" : ids , "title" : titles , "key" : keys} , columns=['id', 'title', 'key']);
    return result;

def main():
    dataFile = "sample_data.csv";
    data = pd.read_csv(dataFile);
    result = getKeywords_textrank(data , 10);
    result.to_csv("keys_texttank.csv" , index = False);

if __name__=='__main__':
    main();
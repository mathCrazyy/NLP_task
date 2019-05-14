#https://www.jianshu.com/p/093ec1eeccff
def filter_word(sentence):
    for uchar in sentence:
        if(uchar>=u'\u4e00' and uchar<=u'\u9fa5'):
            continue
        if(uchar >= u'\u0030' and uchar<=u'\u0039'):
            continue
        if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
            continue
        else:
            sentence=sentence.replace(uchar,"")
    return sentence

x=filter_word("12a你好")
print(x)
import os
base_dir="E:/data/word_nlp/cnews/"
train_dir=os.path.join(base_dir,"cnews.test.txt")
contents=[]
import jieba
def get_content(filename):
    with open(filename, encoding="utf-8")as f:
        for line in f:
            try:
                label, content = line.strip().split(("\t"))
                if content:
                    content = filter_word(content)
                    split_word = " ".join(jieba.lcut(content))
                    contents.append(split_word)
            except:
                pass
    return contents
contents=get_content(train_dir)
from sklearn.feature_extraction.text import CountVectorizer


#https://blog.csdn.net/ustbbsy/article/details/80047916
vectorizer=CountVectorizer(min_df=2)  #min_df指的是全文本中出现的最小频词为min_df，在单个文本中可能出现的次数要低于该数值。
#corpus=["我 来自 大猫 星球","低智 的 人类","忘记 了 生存 的 条件","人类 被 灭亡 ， 忘记 会 发生 在 32 年 后"]
#corpus=["aa b","aa aa cc","aa aa cc"]
corpus=["我们 是 好人 大大 他们 坏人 真的","他们 是 坏人 真的","aa aa cc"]
#https://www.cnblogs.com/yjybupt/p/10437442.html  单个字符会被过滤掉

X=vectorizer.fit_transform(corpus)
#X=vectorizer.fit_transform(contents[0:5000])
print(X)
print(vectorizer.get_feature_names())

bigram_vectorizer=CountVectorizer(ngram_range=(1,2),min_df=1)
analyze=bigram_vectorizer.build_analyzer()
print(analyze("我们 是 好人 真的 坏人"))

X_2=bigram_vectorizer.fit_transform(contents[0:200]).toarray()
#print(X_2)
bi_word_all=bigram_vectorizer.get_feature_names()

file_out=open("bigram.txt","w")
for one in bi_word_all:
    file_out.write(one+"\n")
file_out.close()

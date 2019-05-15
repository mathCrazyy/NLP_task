from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus=["我们 是 天使 的 爸爸 美女","美女 你们 天使 是 天使 的 孩子","美女 你们 做为 恶魔 不会 心痛 嘛","美女 天使 和 恶魔 的 区别 是 什么"]

vectorizer=CountVectorizer()

tranformer=TfidfTransformer()
tfidf=tranformer.fit_transform(vectorizer.fit_transform(corpus))

print(tfidf)

x=zip(vectorizer.get_feature_names(),range(0,len(vectorizer.get_feature_names())))
print(list(x))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2=TfidfVectorizer()
re=tfidf2.fit_transform(corpus)
print(re)

from sklearn import metrics as mr

label=[1,1,1,1,1,0,0,0,0]
x1=["x","x","x","x","x","y","y","y","y"]
x2=["x","x","x","y","y","y","y","x","x"]
x3=["x","x","x","x","x","y","y","y","y"]
x4=["x","x","x","x","y","y","y","y","y"]
res1=mr.mutual_info_score(label,x1)
res2=mr.mutual_info_score(label,x2)
res3=mr.mutual_info_score(label,x3)
res4=mr.mutual_info_score(label,x4)

print(res1)
print(res2)
print(res3)
print(res4)
print("---------------")
res1_4=mr.mutual_info_score(x1,x4)
res1_3=mr.mutual_info_score(x1,x3)
res1_2=mr.mutual_info_score(x1,x2)
print(res1_4)
print(res1_3)
print(res1_2)

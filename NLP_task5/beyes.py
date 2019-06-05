# https://shimo.im/docs/VQJmjTPvQ241lHTf  #作业链接
# <160又50个  160-170 有250个 170+ 有200个
import numpy as np
x1=140+np.random.random_sample([1,50])*20
x2=160+np.random.random_sample([1,250])*10
x3=170+np.random.random_sample([1,200])*20

xx1=140+np.random.random_sample([1,150])*20
xx2=160+np.random.random_sample([1,750])*10
xx3=170+np.random.random_sample([1,600])*20

X=x1.tolist()[0]+x2.tolist()[0]+x3.tolist()[0]+xx1.tolist()[0]+xx2.tolist()[0]+xx3.tolist()[0]
XX=[[one]for one in X]
X=np.array(XX)

print(X.shape)
y1=np.array([1,0,0])
y1_fuzhu=np.zeros((50,1))
yy1=y1+y1_fuzhu
y2=np.array([0,1,0])
y2_fuzhu=np.zeros((250,1))
yy2=y2+y2_fuzhu
y3=np.array([0,0,1])
y3_fuzhu=np.zeros((200,1))
yy3=y3+y3_fuzhu

#y=yy1.tolist()+yy2.tolist()+yy3.tolist()
y=[0]*50+[1]*250+[2]*200+[0]*150+[1]*750+[2]*600
y=np.array(y)
print(y)
print(y.shape)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
#高斯贝叶斯

def train_model_GaussianNB():
    clf3 = GaussianNB()
    clf3.fit(X[499:], y[499:])#训练模型
    predict_labels = clf3.predict(X[0:499])
    print(predict_labels)
 # 预测对了几个？
    n = 0
    for i in range(len(predict_labels)):
        if (predict_labels[i] == y[i]):
            n = n + 1
    print("高斯贝叶斯:")
    # 正确率
    print (n / 499.0)
    # 混淆矩阵
    confusion_matrix(y[0:499], predict_labels)
    return

import string
import random
def generate_doc(album="a ",label="0"):

    all_abc = string.ascii_lowercase
    abc_index = np.random.randint(25, size=100)


    sentence_a = album * 90 + " ".join([all_abc[one] for one in abc_index])
    x = sentence_a.split()
    random.shuffle(x)
    x.append(label)
    return " ".join(x)
sentence=generate_doc()
all_sentence=[]
for i in range(0,260):
    all_sentence.append(generate_doc("a ","0"))
for i in range(0, 260):
    all_sentence.append(generate_doc("b ", "1"))
"""
for i in range(0,5000):
    all_sentence.append(generate_doc("c ","2"))
for i in range(0,5000):
    all_sentence.append(generate_doc("d ","3"))

"""
#print(all_sentence)
random.shuffle(all_sentence)
#print(all_sentence)
X=[]
y=[]
import string
abc=[one for one in string.ascii_lowercase]
dict_abc=zip(abc,range(0,len(abc)))
dict_abc=dict(dict_abc)

for one in all_sentence:
    X.append([dict_abc[small_one] for small_one in one[0:-1].split()])
    y.append(int(one[-1]))

X=np.array(X)
y=np.array(y)




from sklearn.naive_bayes import MultinomialNB

#多项式贝叶斯

def train_model_MultinomialNB():
    print("+++++++++++++++++++++++++++++++++")
    clf = MultinomialNB()
    #训练模型
    print(type(X))
    #print(X.shape)
    print(type(y))
    print(X[0],"---------",y[0])
    print(X[499:].shape)
    clf.fit(X[499:],y[499:])
    #预测训练集
    predict_labels = clf.predict(X[0:499])
    print(predict_labels)
    #for one in predict_labels:
     #   print(one)

    #预测对了几个？
    n = 0
    for i in range(len(predict_labels)):
        if(predict_labels[i] == y[i]):
            n = n + 1
    print("多项式贝叶斯:")
    #正确率
    print (n/499.0)
    #混淆矩阵
    m=confusion_matrix(y[0:499], predict_labels)
    return m

from sklearn.naive_bayes import BernoulliNB

#伯努利贝叶斯

def train_model_BernoulliNB():

    clf2 = BernoulliNB()
    clf2.fit(X[499:], y[499:])
    print(X[0],"---",y[0])
    predict_labels = clf2.predict(X[0:499])
    # 预测对了几个？
    n = 0
    for i in range(len(predict_labels)):
        if (predict_labels[i] == y[i]):
            n = n + 1
    print("伯努利贝叶斯:")
    # 正确率
    print(predict_labels)
    print( n / 499.0)
    # 混淆矩阵
    m= confusion_matrix(y[0:499], predict_labels)
    return m

#train_model_GaussianNB()
m=train_model_MultinomialNB()
#m=train_model_BernoulliNB()
print(m)
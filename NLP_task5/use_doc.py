# https://blog.csdn.net/xiaotian127/article/details/86836571  #数据集放的位置
#https://blog.csdn.net/cymy001/article/details/79052366
from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset="all")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)

from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()
X_train=vec.fit_transform(X_train)
X_test=vec.transform(X_test)
print(type(y_train))
print("y_train: ",y_train)
print(X_train[0])
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
def train_model_MultinomialNB():
    print("+++++++++++++++++++++++++++++++++")
    clf = MultinomialNB()
    #训练模型

    print(X_test[0],"---------",y_test[0])

    clf.fit(X_train,y_train)
    #预测训练集
    predict_labels = clf.predict(X_test)
    print(predict_labels)
    #for one in predict_labels:
     #   print(one)

    #预测对了几个？
    n = 0
    for i in range(len(predict_labels)):
        if(predict_labels[i] == y_test[i]):
            n = n + 1
    print("多项式贝叶斯:")
    #正确率
    print (n/len(y_test))
    #混淆矩阵
    confusion_matrix(y_test, predict_labels)
    return

train_model_MultinomialNB()
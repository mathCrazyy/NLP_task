#回头看： 特征选择：
## https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
#https://www.colabug.com/3730537.html

#参考: https://www.imooc.com/article/48072
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

X,y=datasets.make_classification(
    n_samples=2000,
    n_features=10,#每个样本具有10个特征
    n_informative=4,
    n_redundant=1,
    n_classes=2,
    n_clusters_per_class=1,#每个类别有几个簇构成
    weights=[0.1,0.9],#样本比例
    flip_y=0.1,#应该是造成误差（引入噪声）的意思，样本之间标签交换
    random_state=2019
)
df_all=pd.DataFrame(X)
df_all["y"]=y

pca=PCA(n_components=2)
pca.fit(X)
X_pca=pca.transform(X)

df_X=pd.DataFrame(columns=["pca_a","pca_b","y"])
df_X.pca_a=X_pca[:,0]
df_X.pca_b=X_pca[:,1]
df_X.y=y
print(type(X_pca))

print(X.shape,"  --> ",X_pca.shape)
#sns.lmplot(x="pca_a",y="pca_b",data=df_X,hue="y",fit_reg=False,markers=["o","x"],size=8,aspect=1.5,legend=True)
"""
colormap=plt.cm.RdBu
sns.heatmap(df_all.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor="white",annot=True)
plt.legend(fontsize=20,bbox_to_anchor=(0.98,0.6),edgecolor="r")
plt.xlabel("aixs_1",fontsize=17)

plt.ylabel("aixs_2",fontsize=17)
plt.show()

"""

from sklearn.model_selection import KFold,StratifiedKFold

kf=StratifiedKFold(n_splits=2,random_state=2019)
for train_index,test_index in kf.split(X,y):
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
pos_prob_lr=lr.predict_proba(X_test)[:,1]

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
pos_prob_rf=rf.predict_proba(X_test)[:,1]

import numpy as np
def get_roc(pos_prob,y_true):
    pos=y_true[y_true==1]
    neg=y_true[y_true==0]
    threshold=np.sort(pos_prob)[::-1]
    y=y_true[pos_prob.argsort()[::-1]]
    tpr_all=[0]
    fpr_all=[0]
    tpr=0
    fpr=0
    x_step=1/float(len(neg))
    y_step=1/float(len(pos))
    y_sum=0
    for i in range(len(threshold)):
        if(y[i]==1):
            tpr+=y_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
        else:
            fpr+=x_step
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            y_sum+=tpr
    return tpr_all,fpr_all,y_sum*x_step

#原数据测试
"""
tpr_lr,fpr_lr,auc_lr=get_roc(pos_prob_lr,y_test)

tpr_rf,fpr_rf,auc_rf=get_roc(pos_prob_rf,y_test)
plt.figure(figsize=(10,6))

plt.plot(fpr_lr,tpr_lr,label="Logistic Regression (AUC: {:.3f})".format(auc_lr),linewidth=2)
plt.plot(fpr_rf,tpr_rf,'g',label="Random Forest (AUC: {:.3f})".format(auc_rf),linewidth=2)

plt.xlabel("false_positive_rate")
plt.ylabel("true_positive_rate")
plt.title("roc curve",fontsize=16)
plt.legend(loc="lower right",fontsize=16)
plt.show()
"""




#改变测试比例

X_test_dup = np.vstack((X_test,X_test[y_test==0],X_test[y_test==0],X_test[y_test==0],X_test[y_test==0],
                        X_test[y_test==0],X_test[y_test==0],X_test[y_test==0],X_test[y_test==0],X_test[y_test==0]))
index=np.random.permutation(len(X_test_dup))
X_test_dup=X_test_dup[index]
y_test_dup = np.array(y_test.tolist() + y_test[y_test==0].tolist()*9)
y_test_dup=y_test_dup[index]

pos_prob_lr_dup = lr.predict_proba(X_test_dup)[:,1]
pos_prob_rf_dup = rf.predict_proba(X_test_dup)[:,1]
tpr_lr_dup,fpr_lr_dup,auc_lr_dup = get_roc(pos_prob_lr_dup,y_test_dup)
tpr_rf_dup,fpr_rf_dup,auc_rf_dup = get_roc(pos_prob_rf_dup,y_test_dup)
"""
plt.figure(figsize=(10,6))
plt.plot(fpr_lr_dup,tpr_lr_dup,label="Logistic Regression (AUC: {:.3f})".format(auc_lr_dup),linewidth=2)
plt.plot(fpr_rf_dup,tpr_rf_dup,'g',label="Random Forest (AUC: {:.3f})".format(auc_rf_dup),linewidth=2)
plt.xlabel("False Positive Rate",fontsize=16)
plt.ylabel("True Positive Rate",fontsize=16)
plt.title("ROC Curve",fontsize=16)
plt.legend(loc="lower right",fontsize=16)
plt.show()

"""


def get_pr(pos_prob,y_true):
    pos = y_true[y_true==1]
    threshold = np.sort(pos_prob)[::-1]
    y = y_true[pos_prob.argsort()[::-1]]
    recall = [] ; precision = []
    tp = 0 ; fp = 0
    auc = 0
    for i in range(len(threshold)):
        if y[i] == 1:
            tp += 1
            recall.append(tp/len(pos))
            precision.append(tp/(tp+fp))
            auc += (recall[i]-recall[i-1])*precision[i]
        else:
            fp += 1
            recall.append(tp/len(pos))
            precision.append(tp/(tp+fp))
    return precision,recall,auc

precision_lr,recall_lr,auc_lr = get_pr(pos_prob_lr,y_test)
precision_rf,recall_rf,auc_rf = get_pr(pos_prob_rf,y_test)

plt.figure(figsize=(10,6))
plt.plot(recall_lr,precision_lr,label="Logistic Regression (AUC: {:.3f})".format(auc_lr),linewidth=2)
plt.plot(recall_rf,precision_rf,label="Random Forest (AUC: {:.3f})".format(auc_rf),linewidth=2)
plt.xlabel("Recall",fontsize=16)
plt.ylabel("Precision",fontsize=16)
plt.title("Precision Recall Curve",fontsize=17)
plt.legend(fontsize=16)
plt.show()


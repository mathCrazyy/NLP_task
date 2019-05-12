import tensorflow as tf
from tensorflow import  keras
import numpy as np

print(tf.__version__)
#数据加载
imdb=keras.datasets.imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
#数据观察---------------------
print("imdb训练数据原格式: {},labels:{}".format(len(train_data),len(train_labels)))
print(train_data[0])
print(train_data.shape)
print(len(train_data[0]))
print(len(train_data[1]))
all_len=[len(one) for one in train_data]
max_len=max(np.array(all_len))
print("最长句子的len: ",max_len)#最长的为2494
from numpy import *
print("中位数",median(all_len))#中位数为178
bin_count=np.bincount(all_len)
#print("众数",bin_count[max(bin_count)])
print("众数",np.argmax(bin_count))
#数据观察---------------------


word_index=imdb.get_word_index()
word_index={k:(v+3)for k,v in word_index.items()}
word_index["<PAD>"]=0
word_index["SATART"]=1
word_index["UNK"]=2
word_index["UNUSED"]=3
reverse_word_index=dict([value,key]for (key,value)in word_index.items())

def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?")for i in text])
print(decode_review(train_data[0]))
print(decode_review(train_data[1]))

#数据准备，目前整数数组需要转为张良才能送入神经网络中。
"""
1.独热编码，需要转为0和1构成的向量，那么就会需要一个num_words*num_review是（单词个数*评论条目个数）
2.填充数组，是他们具有相同的长度（句子长度相同），那么需要一个max_length*num_reviews（句子最大长度*评论条目个数）
直观上，阔以看出第二种方法所占用的内存，是小于第一种的。
"""
#虽然众数是132，不知道为啥取了256
train_data=keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=178
)
test_data=keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=178
)
print(train_data[0].shape)
print(train_data[0])#阔以看出后面所跟的0为后补的。

#构建模型

vocab_size=10000

model=keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,16))# 16阔以改变(batch, sequence, embedding)
#odel.add(keras.layers.GlobalAveragePooling1D())m
model.add(keras.layers.GlobalMaxPool1D())
model.add(keras.layers.Dense(8,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))

model.summary()

#损失函数和优化器
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss="binary_crossentropy",
              metrics=["accuracy"])

x_val=train_data[:10000]
partial_x_train=train_data[10000:]
y_val=train_labels[:10000]
partial_y_train=train_labels[10000:]

#模型训练
history=model.fit(partial_x_train,partial_y_train,epochs=40,batch_size=512,validation_data=(x_val,y_val),verbose=1)

#评估模型
print("模型评估....")
results=model.evaluate(test_data,test_labels)

print(results)

history_dict=history.history
history_dict.keys()

import matplotlib.pyplot as plt
acc=history.history["acc"]
val_acc=history.history["val_acc"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]

epochs=range(1,len(acc)+1)

# bo 是蓝色点的曲线
# b 是蓝色线的线
"""
plt.plot(epochs,loss,"bo",label="training loss")
plt.plot(epochs,val_loss,"b",label="validation loss")

plt.title("training and validation loss")
plt.xlabel("eopchs")
plt.ylabel("loss")
plt.legend()
plt.show()

"""
plt.plot(epochs,acc,"bo",label="training_acc")
plt.plot(epochs,val_acc,"b",label="val_acc")
plt.title("training and validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()







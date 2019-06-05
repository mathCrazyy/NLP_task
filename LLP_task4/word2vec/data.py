#https://shimo.im/docs/VQJmjTPvQ241lHTf
import os
import tensorflow as tf
import collections
import numpy as np
import random

class Dataset(object):
    def __init__(self,file):
        self.file=file
        self.data_index=0
        self._build_dataset()

    def _build_dataset(self):
        if not os.path.exists(self.file):
            raise ValueError("file does not exist-->%s"%self.file)
        f=open(self.file,mode='rt',encoding='utf8')
        self.data=tf.compat.as_str(f.read()).split()
        if f:
            f.close()
        c=collections.Counter(self.data).most_common()
        self.vocab_size=len(c)
        self.counter=c.insert(0,("UNK",-1))
        self.vocab_size+=1
        self.word2id=dict()
        self.id2word=dict()

        for word,_ in c:
            self.word2id[word]=len(self.word2id)
            self.id2word[len(self.id2word)]=word
        def gen_batch_inputs(self,batch_size,skip_window):
            raise NotImplementedError()

class skipGramDataset(Dataset):
    def gen_batch_inputs(self,batch_size,window_size):
        features=np.ndarray(shape=(batch_size,),dtype=np.int32)
        labels=np.ndarray(shape=(batch_size,),dtype=np.int32)
        i=0
        while True:
            if self.data_index==len(self.data):
                self.data_index=0
            left=max(0,self.data_index-window_size)










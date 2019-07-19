import pickle as pkl
import gzip
import numpy
import random
import math

symbol={'O':0,'B':1,'M':2,'E':3,'S':4}

class TextIterator:
    """Simple text iterator."""
    def __init__(self,source,target,hparams,batch_size,shuffle=True):
        self.source=open(source,'r')
        self.target=open(target,'r')
        self.hparams=hparams
        self.source_buffer = []
        self.target_buffer = []
        random.seed(2018)
        self.batch_size=batch_size
        self.k = hparams.batch_size * 20
        self.end_of_data=False
        self.shuffle=shuffle
        

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
        
    """return a batch"""
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())


            if self.shuffle:
                index=list(range(len(self.source_buffer)))
                random.shuffle(index)
                self.source_buffer=[self.source_buffer[i] for i in index]
                self.target_buffer=[self.target_buffer[i] for i in index]
                
        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        for i in range(self.batch_size):
            try:
                ss = self.source_buffer.pop(0)
                tt = self.target_buffer.pop(0)
                assert len(ss)==len(tt)
                source.append(ss)
                target.append(tt)
            except:
                break


        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        length,source, target=self.preprocess(source, target)
        
        return source, target,length
    
    """compute a batch"""
    def preprocess(self,source,target):
        length=[]
        for s in source:
            length.append(len(s))
            
        #padding data to a batch 
        for i in range(len(source),self.batch_size):
            source.append(['<pad>'])
            target.append(['O'])
            length.append(0)
            
        #padding data to max length
        max_length=max(length)
        for i in range(self.batch_size):
            source[i]+=['<pad>']*(max_length-len(source[i]))
            target[i]+=['O']*(max_length-len(target[i]))
            
        #conver string label to idx label
        for i in range(self.batch_size):   
            for j in range(max_length):
               target[i][j]=symbol[target[i][j]] 
        
        return length,source, target







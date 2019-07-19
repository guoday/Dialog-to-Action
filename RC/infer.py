import json
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import RC.main as model


def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)

 
class Predicate_model(object):
    def __init__(self,path): 
        self.sess,self.infer_model=model.infer(path)
            
    def predicate_infer(self,sentences):
        """Relation classify
        Args:
            sentences: a list of sentence
        Returns:
            A list of relations appear in each sentence with top2 scores
        """        
        s=[]
        length=[]
        label=[]   
        for temp in sentences:
            temp=tokenize(temp).split()
            s.append(temp)
            length.append(len(temp))
            label.append('0')
        max_length=max(length)
        for i in range(len(s)):
            s[i]+=['<pad>']*(max_length-len(s[i]))
        pred=self.infer_model.infer(self.sess,[s,label,length])
        return [[x.decode('utf-8') for x in p] for p in pred]                       
                


        

"""Create a class to linking entity"""
import json
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import EDL.main as model

symbol={0:'O',1:'B',2:'M',3:'E',4:'S'}
def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)



class EL(object):
    def __init__(self,keep_num,path): 
        #reload the dictionary
        #key:a string 
        #values:list of ids of entities with highest score
        self.inverse_index=json.load(open('data/EDL/inverse_index.json','r'))
        
        #reload infer model
        self.sess,self.infer_model=model.infer(path)
        
    def entity_detection(self,sentences):
        """Entity detection
        Args:
            sentences: a list of sentence
        Returns:
            A list of list of entities appear in each sentence
        """
       
        s=[]
        length=[]
        label=[]
        for temp in sentences:
            temp=tokenize(temp).split()
            s.append(temp)
            length.append(len(temp))
        max_length=max(length)
        for i in range(len(s)):
            s[i]+=['<pad>']*(max_length-len(s[i]))
            label.append([0]*max_length)
            
        pred=self.infer_model.infer(self.sess,[s,label,length])
        pred=[[symbol[i] for i in w] for w in pred]
        
        entities=[]
        for i in range(len(pred)):
            candidate=[]
            e=[]
            flag=False
            for j in range(length[i]):
                if pred[i][j]=="B":
                    e.append(s[i][j])
                    flag=True
                elif pred[i][j]=="S":
                    e=[]
                    flag=False
                    candidate.append(s[i][j])
                elif pred[i][j]=="M":
                    e.append(s[i][j])
                    pass
                elif pred[i][j]=="O":
                    e=[]
                    flag=False
                elif pred[i][j]=="E":
                    e.append(s[i][j])
                    candidate.append(' '.join(e))  
                    e=[]
                    flag=False
            entities.append(candidate)                          
                
        return entities
    
    def entity_linking(self,entities_list):
        """Entity linking
        Args:
            entities_list: a list of entities string
        Returns:
            A list of entities ids with highest scores
        """
        results=[]
        for entities in entities_list:
            result=[]
            for e in entities:
                if tokenize(e) in self.inverse_index:
                    result+=self.inverse_index[tokenize(e)]['idxs']
            results.append(list(set(result)))
        return results
        

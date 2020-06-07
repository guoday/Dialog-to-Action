#!/usr/bin/env python3
#-*-coding=utf-8-*-  
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import json
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pickle
import numpy
import random
from tqdm import tqdm
import BFS.agent as agent
import random
import BFS.parser as Parser
import threading
import multiprocessing
import time
import timeout_decorator
import argparse

def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)

class Memory(object):
    """ Dialog Memory
    keep and update entities and relation (in our code, we also use predicate(pre) to represent relation)
    """
    def __init__(self):
        self.entities=[]
        self.pre=[]
    def current_state(self,entities,pre):
        current_entities=entities+self.entities
        if pre:
            current_pre=pre
        else:
            current_pre=self.pre
        return current_entities,current_pre
    def update(self,entities,pre):
        self.entities=entities
        self.pre=pre
            
    def clear(self):
        self.entities=[]
        self.pre=[]        



def get_data():
    #load dataset
    dir="data/BFS/"
    train=[]
    dev=[]
    test=[]
    cover_num={}
    for root, dirs, files in os.walk(dir):
        for file in files:
            temp=os.path.join(root,file)
            if '.json' in temp:
                if 'train' in temp:
                    train.append(temp)

    return train



def parsing(files,cover_num_True,cover_num_False,verb,beam_size):
    parser=Parser.Parser(database)
    memory=Memory()    
    for f in tqdm(files,total=len(files)):
        #load dataset
        dicts=json.load(open(f,'r')) 
        #reset memory
        memory.clear()
        for i in range(0,len(dicts),2): 
            #Extract entity and relation, in BFS, we use entities and relations offered by training dataset
            #In D2A, we only use entities by entity linking and relations by relation classfier  
            #In our setting, we assume that entities and relations are unseen in test dataset 
            if 'entities_in_utterance' in dicts[i]:
                user_entities=dicts[i]['entities_in_utterance']
            else:
                user_entities=[]
            if 'entities_in_utterance' in dicts[i+1]:
                system_entities=dicts[i+1]['entities_in_utterance']
            else:
                system_entities=[]
            if 'relations' in dicts[i]:
                pres=dicts[i]['relations']
            else:
                pres=[]
            if 'type_list' in dicts[i]:
                types=dicts[i]['type_list']
            else:
                types=[]
            numbers=[]
            for x in dicts[i]['utterance'].split():
                try:
                    numbers.append(int(x))
                except:
                    continue
            numbers=list(set(numbers))                   
            entities,pres=memory.current_state(user_entities,pres)

            #Extract answer
            answer=parser.parsing_answer(dicts[i+1]['all_entities'],dicts[i+1]['utterance'],dicts[i]['question-type'])
            try:
                logical_forms,candidate_answers,logical_action=parser.BFS(entities,pres,types,numbers,beam_size)
            except timeout_decorator.TimeoutError:
                logical_forms=[]
                candidate_answers=[]
                logical_action=[]
            #update memory and keep right logical forms and action sequences
            memory.update(user_entities+system_entities,pres)
            True_lf=[]
            True_lf_action=[]
            All_lf=[]
            for item in zip(logical_forms,candidate_answers,logical_action): 
                pred=item[1]
                All_lf.append(item[0])
                All_lf.append((item[0],item[2]))
                if type(pred)==int:
                    pred=[pred]
                if answer==pred:
                    True_lf.append(item[0])
                    True_lf_action.append((item[0],item[2]))

            #eval oracle
            if dicts[i]["question-type"] not in cover_num_True:
                cover_num_True[dicts[i]["question-type"]]=0.0
                cover_num_False[dicts[i]["question-type"]]=0.0
            if len(True_lf_action)!=0:
                cover_num_True[dicts[i]["question-type"]]+=1
            else:
                cover_num_False[dicts[i]["question-type"]]+=1
            dicts[i+1]["true_lf"]=True_lf_action
            dicts[i+1]['all_lf']=All_lf
        
        json.dump(dicts,open(f,'w')) 
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BFS/run.py')   
    parser.add_argument('-mode', required=True,help="offline or online")  
    parser.add_argument('-num_parallel', type=int, default=1,help="degree of parallelly")
    parser.add_argument('-beam_size', type=int, default=1000,help="beam size of BFS")
    parser.add_argument('-max_train', type=int, default=60000,help="number of dialogs to search")
    opt = parser.parse_args()
    #load knowledge base
    global database
    database=agent.KB(mode=opt.mode)
    #load dataset
    thread_num=opt.num_parallel
    train=get_data()
    manager = multiprocessing.Manager()  
                    
    files=train[:opt.max_train]         
    if opt.max_train!=0:
        #allocating task to different thread
        thread_files=[[] for i in range(thread_num)]
        threads=[]
        for idx,f in enumerate(files):
            thread_files[idx%thread_num].append(f)
            
        #to eval oracle of BFS    
        cover_num_True=[manager.dict() for i in range(thread_num)] 
        cover_num_False=[manager.dict() for i in range(thread_num)] 
        
        for i in range(thread_num):
            thread = multiprocessing.Process(target=parsing,args=(thread_files[i], cover_num_True[i],cover_num_False[i], i==0,opt.beam_size))
            thread.start()
            threads.append(thread) 
        for t in threads:
            t.join()
            
        #print result
        cover_num_True=[dict(cover_num_True[i]) for i in range(thread_num)] 
        cover_num_False=[dict(cover_num_False[i]) for i in range(thread_num)]
        cover={}
        for it in cover_num_True[0]:
            cover[it]=[0.0,0.0]
        for i in range(thread_num):
            for it in cover_num_True[i]:
                try:
                    cover[it][0]+=cover_num_False[i][it]
                    cover[it][1]+=cover_num_True[i][it]
                except:
                    pass
                
        for it in cover:
            print(it,round(cover[it][1]*100.0/sum(cover[it]),2))

                
                


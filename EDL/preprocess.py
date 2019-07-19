#!/usr/bin/env python3
#-*-coding=utf-8-*-  
import os
import json
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pickle
import random

def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)


def replace_EO(sentence,EO,entities):
    s=' '.join(sentence)
    e=' '.join(entities)
    s=s.replace(e,' '.join(['XXX']*len(entities)))
    s=s.split()
    assert len(s)==len(EO)
    flag=True
    cont=0
    for i in range(len(s)):
        if s[i]=='XXX':
            s[i]='YYY'
            if flag:
                cont+=1
                flag=False
                if len(entities)==1:
                    EO[i]='S'
                    flag=True
                else:
                    EO[i]='B'
                
            else: 
                cont+=1
                if cont==len(entities):
                    EO[i]='E'
                    flag=True
                else:
                    EO[i]='M'
    return s,EO

def generate_EO(sentence,entities_in_utterance):
    s=sentence.split()
    EO=['O' for i in s]
    for e in entities_in_utterance:
        s,EO=replace_EO(s,EO,e.split())
    return ' '.join(EO)


if __name__ == "__main__":
    #load dataset
    train=[]
    dev=[]
    for root, dirs, files in os.walk("data/CSQA"):
        for file in files:
            temp=os.path.join(root,file)
            if '.json' in temp:
                if 'train' in temp:
                    train.append(temp)
                elif 'valid' in temp:
                    dev.append(temp)
    
    #load entities of knowledge base, and tokenize entities string 
    dict_e=json.load(open('data/kb/items_wikidata_n.json','r'))
    for idx in dict_e:
        dict_e[idx]=tokenize(dict_e[idx])


    #generate training data to folder data/EDL
    #labeling entity of sentences with start(S), begin(B), end(E) and middle(M) symbols 
    for files in [('data/EDL/train',train),('data/EDL/dev',dev)]:
        fin=open(files[0]+'.in','w')
        fout=open(files[0]+'.out','w')
        for f in files[1]:
            f=open(f,'r')
            dicts=json.load(f)
            for d in dicts:
                temp=[]
                if 'entities_in_utterance' in d:
                    temp=d['entities_in_utterance']
                    temp.sort(key=lambda x: -len(x))
                    temp=[dict_e[idx] for idx in temp]
                utterance=tokenize(d["utterance"])
                EO=generate_EO(utterance,temp)
                assert len(utterance.split())==len(EO.split())
                fin.write(utterance+'\n')
                fout.write(EO+'\n')
                
                
    #build vocabulary    
    source_files=['data/EDL/train.in']    
    source={}
    for i in source_files:
        with open(i,'r') as f:
            for line in f:
                line=line.strip().split()
                for word in line:
                    try:
                        source[word]+=1
                    except:
                        source[word]=1
    source=list(source.items())                      
    source.sort(key=lambda x:-x[1])
    source=[('<unk>',100),('<pad>',100)]+source
    with open('data/EDL/vocab.in','w') as f:
        for word in source:
            if word[1]>1:
                f.write(word[0]+'\n')

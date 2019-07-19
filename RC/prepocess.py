#!/usr/bin/env python3
#-*-coding=utf-8-*-  
import os
import json
from nltk.tokenize import WordPunctTokenizer
import json
import numpy as np
import pickle
import numpy
import random

def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)

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



    #generate training data to folder data/RC
    context_size=1
    for files in [('data/RC/train',train),('data/RC/dev',dev)]:
        fin=open(files[0]+'.in','w')
        fout=open(files[0]+'.out','w')
        for f in files[1]:
            f=open(f,'r')
            dicts=json.load(f)
            utterence=["<empty_context>"]*context_size
            for d in dicts:
                if d['speaker']=='USER':
                    utterence.append(tokenize(d["utterance"]))
                    if "relations" in d and len(d["relations"])!=0:
                        for r in d["relations"]:
                            fin.write(' ||| '.join(utterence[-context_size:])+'\n')
                            fout.write(r+'\n')

    #building encoder vocabulary
    source_files=['data/RC/train.in']    
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
    source.sort(key=lambda x: -x[1])
    source=[('<unk>',100),('<pad>',100)]+source
    with open('data/RC/vocab.in','w') as f:
        for word in source:
            if word[1]>1:
                f.write(word[0]+'\n')
                
    #building decoder vocabulary
    source_files=['data/RC/train.out']    
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
    source.sort(key=lambda x: -x[1])
    source=[('<unk>',100)]+source
    with open('data/RC/vocab.out','w') as f:
        for word in source:
            if word[0]!='|||':
                f.write(word[0]+'\n')     

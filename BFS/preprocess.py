#!/usr/bin/env python3
#-*-coding=utf-8-*-  
import sys
import os
import json
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pickle
import numpy
import random
from tqdm import tqdm
from EDL import entity_linking 
from RC import infer
import BFS.agent as agent
def get_id(idx):
    return int(idx[1:])

def preprocess_data():
    """
    For each example, we obtain entities and relation by EDL and RC
    """
    print("Preprocessing data...")
    #load dataset
    train=[]
    dev=[]
    test=[]
    for root, dirs, files in os.walk("data/CSQA"):
        for file in files:
            temp=os.path.join(root,file)
            if '.json' in temp:
                if 'train' in temp:
                    train.append(temp)
                elif 'valid' in temp:
                    dev.append(temp)
                elif 'test' in temp:
                    test.append(temp)
                
    #entity linking and relation predication
    #train=random.sample(train,min(60000,len(train)))
    entity_linking_model=entity_linking.EL(1,'EDL/model')
    predicate_infer_model=infer.Predicate_model('RC/model')
    for files in [('train',train),('dev',dev),('test',test)]:
        cont=0
        for f in tqdm(files[1],total=len(files[1])):
            dicts=json.load(open(f,'r'))
            sentences=[]
            for d in dicts:
                sentences.append(d['utterance'])
            if len(sentences)==0:
                continue
            entities=entity_linking_model.entity_linking(entity_linking_model.entity_detection(sentences))
            predicates=predicate_infer_model.predicate_infer(sentences)

            for i in range(len(dicts)):
                dicts[i]['entities_linking']=entities[i]
                dicts[i]['predicate_prediction']=predicates[i]
            json.dump(dicts,open('data/BFS/'+files[0]+'/QA'+str(cont)+'.json','w'))  
            cont+=1

def create_entity_type():
    print("Creating entity_type dictionary...")
    '''
    Build a dictionary
    key: ids of entity
    values: ids of type
    '''

    dic=json.load(open('data/kb/par_child_dict.json'))
    max_id=0
    for d in tqdm(dic,total=len(dic)):
        for idx in dic[d]:
                max_id=max(max_id,get_id(idx))

    type_dict =['' for i in range(max_id+1)]
    for d in dic:
        for idx in dic[d]:
                type_dict[get_id(idx)]=d
    pickle.dump(type_dict,open('data/BFS/type_kb.pkl','wb'))
    
    return type_dict

def create_relation_type(type_dict,path):
    print("Creating relation_type dictionary...")
    '''
    Build a dictionary
    key: ids of relation
    values: set of type that there's a entity belong to this type and link to the relation (key)
    '''
    dic={}
    for f in path:
        dic_kb=json.load(open(f,'r'))
        for idx in tqdm(dic_kb,total=len(dic_kb)):
            try:
                idx_type=type_dict[get_id(idx)]
            except:
                continue
            for p in dic_kb[idx]:
                if p not in dic:
                    dic[p]=set()
                dic[p].add(idx_type)    
                for y in dic_kb[idx][p]: 
                    try:
                        y_type=type_dict[get_id(y)]
                    except:
                        continue  
                    dic[p].add(y_type) 
    pickle.dump(dic,open('data/BFS/type_predicate_link.pkl','wb'))
    
def create_type_relation_type(type_dict,paths):
    print("Creating type_relation_type dictionary...")
    '''
    Build a dictionary
    key: type _x,direction _t, relation _r, type _y
    values: set of entity ids _e
    information abot this: 
    if direction _t="obj" , return all entity ids _e with having the triple (entity _e with _x type, relation(_r), any one of entity with _y type) 
    if direction _t="sub" , return all entity ids _e with having the triple (any one of entity with _y type, relation(_r), entity _e with _x type) 
    '''
    dic={}
    for f in paths:
        dic_kb=json.load(open(f[0],'r'))
        obj=f[1]
        sub=f[2]
        for idx in tqdm(dic_kb,total=len(dic_kb)):
            try:
                idx_type=type_dict[get_id(idx)]
            except:
                continue
            if idx_type=='':
                continue
            if idx_type not in dic:
                dic[idx_type]={}
                dic[idx_type][obj]={}
                dic[idx_type][sub]={}
            for p in dic_kb[idx]:
                if (obj,p,idx_type) not in dic:
                    dic[(obj,p,idx_type)]=set()
                dic[(obj,p,idx_type)].add(idx)
                if p not in dic[idx_type][sub]:
                    dic[idx_type][sub][p]={}
                for y in dic_kb[idx][p]: 
                    try:
                        y_type=type_dict[get_id(y)]
                    except:
                        continue
                    if y_type=="":
                        continue
                    if y_type not in dic[idx_type][sub][p]:
                        dic[idx_type][sub][p][y_type]={}
                    if y not in dic[idx_type][sub][p][y_type]:
                        dic[idx_type][sub][p][y_type][y]=set()
                    if idx!=y:
                        dic[idx_type][sub][p][y_type][y].add(idx)
                    if y_type not in dic:
                        dic[y_type]={}
                        dic[y_type][obj]={}
                        dic[y_type][sub]={}  
                    if p not in dic[y_type][obj]:
                        dic[y_type][obj][p]={}
                    if idx_type not in dic[y_type][obj][p]:
                        dic[y_type][obj][p][idx_type]={}
                    if idx not in dic[y_type][obj][p][idx_type]:
                        dic[y_type][obj][p][idx_type][idx]=set()
                    if idx!=y:
                        dic[y_type][obj][p][idx_type][idx].add(y)  
    for x in dic:
        if type(x)==tuple:
            continue
        for p in dic[x]['sub']:
            for y in dic[x]['sub'][p]:
                temp=[]
                for idx in dic[x]['sub'][p][y]:
                    temp.append((idx,dic[x]['sub'][p][y][idx]))
                dic[x]['sub'][p][y]=temp

    for x in dic:
        if type(x)==tuple:
            continue
        for p in dic[x]['obj']:
            for y in dic[x]['obj'][p]:
                temp=[]
                for idx in dic[x]['obj'][p][y]:
                    temp.append((idx,dic[x]['obj'][p][y][idx]))
                dic[x]['obj'][p][y]=temp

    pickle.dump(dic,open('data/BFS/pre_type.pkl','wb'))
    
def preprocess_kb():

    type_dict=create_entity_type()
    create_relation_type(type_dict,["data/kb/wikidata_short_1.json","data/kb/wikidata_short_2.json",
                                    "data/kb/comp_wikidata_rev.json"])
    create_type_relation_type(type_dict,[["data/kb/wikidata_short_1.json","obj","sub"],["data/kb/wikidata_short_2.json","obj","sub"],["data/kb/comp_wikidata_rev.json","sub","obj"]])

    

if __name__ == "__main__":
    #preprocess_data
    preprocess_data()

    #preprocess data format of knowledge base            
    preprocess_kb()   
    
    #create knowledge base
    print("Create knowlege base...")
    agent.create_kb('data/kb')



import os
import requests
from urllib import request
import pickle
import json
def get_id(idx):
    return int(idx[1:])

def create_kb(path):
    entity_items=json.load(open(os.path.join(path, "items_wikidata_n.json"),'r'))
    max_id=0
    for idx in entity_items:
        max_id=max(max_id,get_id(idx))

    graph =[{} for i in range(max_id+1)]  
    cont=0
    for idx in entity_items:
        graph[get_id(idx)]['name']=entity_items[idx]
        
        
    sub_predict_obj=json.load(open(os.path.join(path, "wikidata_short_1.json"),'r'))
    for idx in sub_predict_obj:
            for x in sub_predict_obj[idx]:
                sub_predict_obj[idx][x]=set(sub_predict_obj[idx][x])
            graph[get_id(idx)]['sub']=sub_predict_obj[idx]

    sub_predict_obj=json.load(open(os.path.join(path, "wikidata_short_2.json"),'r'))   
    for idx in sub_predict_obj:
            for x in sub_predict_obj[idx]:
                sub_predict_obj[idx][x]=set(sub_predict_obj[idx][x])        
            graph[get_id(idx)]['sub']=sub_predict_obj[idx]   

    obj_predict_sub=json.load(open(os.path.join(path, "comp_wikidata_rev.json"),'r'))
    for idx in obj_predict_sub:
            for x in obj_predict_sub[idx]:
                obj_predict_sub[idx][x]=set(obj_predict_sub[idx][x])
            graph[get_id(idx)]['obj']=obj_predict_sub[idx] 
    pickle.dump(graph,open('data/BFS/wikidata.pkl','wb'))
class KB(object):
    def __init__(self,mode='online'): 
        if mode!='online':
            if not os.path.exists('data/BFS/wikidata.pkl'):
                create_kb('data/kb')
            self.graph=pickle.load(open('data/BFS/wikidata.pkl','rb'))
            self.type_dict=pickle.load(open('data/BFS/type_kb.pkl','rb'))
            self.pre_type=pickle.load(open('data/BFS/pre_type.pkl','rb'))
        else:
            self.graph=None
            self.type_dict=None
            self.pre_type=None
            
    def sub_pre(self,sub,pre):
        if self.graph is not None:
            if 'sub' in self.graph[get_id(sub)] and pre in self.graph[get_id(sub)]['sub']:
                return self.graph[get_id(sub)]['sub'][pre]
            else:
                return None
        else:
            json_pack = dict() 
            json_pack['op']="sub_pre"
            json_pack['sub']=sub
            json_pack['pre']=pre
            content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']
            if content is not None:
                content=set(content)            
            return content
        
    def obj_pre(self,obj,pre):
        if self.graph is not None:
            if 'obj' in self.graph[get_id(obj)] and pre in self.graph[get_id(obj)]['obj']:
                return self.graph[get_id(obj)]['obj'][pre]
            else:
                return None
        else:
            json_pack = dict() 
            json_pack['op']="obj_pre"
            json_pack['obj']=obj
            json_pack['pre']=pre
            content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']
            if content is not None:
                content=set(content)
            return content
        
    def entity_link_predicate(self,entity,pre):
        if self.graph is not None:
            g=self.graph[get_id(entity)]
            return ('sub' in g and pre in g['sub']) or ('obj' in g and pre in g['obj'])
        else:
            json_pack = dict() 
            json_pack['op']="entity_link_predicate"
            json_pack['entity']=entity
            json_pack['pre']=pre
            content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']          
            return content    
        
    def entity_type(self,entity):
        if self.type_dict is not None:
            try:
                return self.type_dict[get_id(entity)]
            except:
                return "empty"
        else:
            json_pack = dict() 
            json_pack['op']="entity_type"
            json_pack['entity']=entity
            content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']
            return content   
        
    def pre_sub_type(self,sub_type,pre,obj_type):
        if self.pre_type is not None:
            try:
                res=self.pre_type[sub_type]['sub'][pre][obj_type]
                all_e=self.pre_type[('sub',pre,obj_type)]-set([x[0] for x in res])
                _set=set()
                for e in all_e:
                    res.append((e,_set))
                return res
            except:
                return None
        else:
            json_pack = dict() 
            json_pack['op']="pre_sub_type"
            json_pack['pre']=pre
            json_pack['sub']=sub_type
            json_pack['obj']=obj_type
            content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']
            if content is not None:
                content=[(x[0],set(x[1])) for x in content]
            return content            
        
    def pre_obj_type(self,obj_type,pre,sub_type):
        if self.pre_type is not None:
            try:
                res=self.pre_type[obj_type]['obj'][pre][sub_type]
                all_e=self.pre_type[('obj',pre,sub_type)]-set([x[0] for x in res])
                _set=set()
                for e in all_e:
                    res.append((e,_set))
                return res
            except:
                return None
        else:
            json_pack = dict() 
            json_pack['op']="pre_obj_type"
            json_pack['pre']=pre
            json_pack['obj']=obj_type
            json_pack['sub']=sub_type
            content=requests.post("http://127.0.0.1:5000/post",json=json_pack).json()['content']
            if content is not None:
                content=[(x[0],set(x[1])) for x in content]
            return content         
        
        
        
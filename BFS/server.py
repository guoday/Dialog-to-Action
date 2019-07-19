import os
import json
import pickle
from flask import Flask, request, jsonify
app = Flask(__name__)

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

    
def sub_pre(sub,pre):
    if 'sub' in graph[get_id(sub)] and pre in graph[get_id(sub)]['sub']:
        return list(graph[get_id(sub)]['sub'][pre])
    else:
        return None
    
def obj_pre(obj,pre):
    if 'obj' in graph[get_id(obj)] and pre in graph[get_id(obj)]['obj']:
        return list(graph[get_id(obj)]['obj'][pre])
    else:
        return None
    
def entity_link_predicate(entity,pre):
    g=graph[get_id(entity)]
    return ('sub' in g and pre in g['sub']) or ('obj' in g and pre in g['obj'])

def entity_type(entity):
    return type_dict[get_id(entity)]

def pre_sub_type(sub_type,pre,obj_type):
    try:
        res=[(x[0],list(x[1])) for x in pre_type[sub_type]['sub'][pre][obj_type]]
        all_e=pre_type[('sub',pre,obj_type)]-set([x[0] for x in res])
        for e in all_e:
            res.append((e,[]))
        return res
    except:
        return None
           
def pre_obj_type(obj_type,pre,sub_type):
    try:
        res=[(x[0],list(x[1])) for x in pre_type[obj_type]['obj'][pre][sub_type]]
        all_e=pre_type[('obj',pre,sub_type)]-set([x[0] for x in res])
        for e in all_e:
            res.append((e,[]))
        return res
    except:
        return None 

@app.route('/post', methods = ['POST'])
def post_res():
    response={}
    jsonpack = request.json
    if jsonpack['op']=="sub_pre":
        response['content']=sub_pre(jsonpack['sub'],jsonpack['pre'])
    elif jsonpack['op']=="obj_pre":
        response['content']=obj_pre(jsonpack['obj'],jsonpack['pre'])
    elif jsonpack['op']=="entity_link_predicate":
        response['content']=entity_link_predicate(jsonpack['entity'],jsonpack['pre'])
    elif jsonpack['op']=="entity_type":
        response['content']=entity_type(jsonpack['entity'])
    elif jsonpack['op']=="pre_sub_type":
        response['content']=pre_sub_type(jsonpack['sub'],jsonpack['pre'],jsonpack['obj'])
    elif jsonpack['op']=="pre_obj_type":
        response['content']=pre_obj_type(jsonpack['obj'],jsonpack['pre'],jsonpack['sub'])    
    return jsonify(response)
    

if __name__ == '__main__':
    if not os.path.exists('data/BFS/wikidata.pkl'):
        create_kb('data/kb')
    global graph
    graph=pickle.load(open('data/BFS/wikidata.pkl','rb'))
    global type_dict
    type_dict=pickle.load(open('data/BFS/type_kb.pkl','rb'))
    global pre_type
    pre_type=pickle.load(open('data/BFS/pre_type.pkl','rb'))
    app.run(host='127.00.0.1', port=5000, use_debugger=True)

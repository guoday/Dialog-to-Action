import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from SMP import data_iterator 
import BFS.parser as Parser
import json
import pickle
import BFS.agent as agent
import os
import random
from tqdm import tqdm
import argparse
def create_hparams():
    return {'batch_size':1, ##don't modify
           }

def get_data():
    #load training dataset
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

types_dict=json.load(open('data/kb/child_par_dict_name_2_corr.json'))
def match_lf_entity(lf_1,lf_2):
    t1=[]
    t2=[]
    for l in lf_1:
        if type(l)==str and l.startswith('Q') and  l not in types_dict:
            t1.append('e')
        else:
            t1.append(l)
    for l in lf_2:
        if type(l)==str and l.startswith('Q') and  l not in types_dict:
            t2.append('e')  
        else:
            t2.append(l)
    if len(t2)+1!=len(t1):
        return None
    for i in range(2):
        if i+len(t2)>len(t1):
            return None
        if t2==t1[i:i+len(t2)]:
            return lf_1[:i]+['copy',(lf_1[i:i+len(t2)])]+lf_1[i+len(t2):] 
    return None    
    
    

def match_lf_pred(lf_1,lf_2):
    t1=[]
    t2=[]
    for l in lf_1:
        if type(l)==str and l.startswith('P'):
            t1.append('r')
        else:
            t1.append(l)
    for l in lf_2:
        if type(l)==str and l.startswith('P'):
            t2.append('r')
        else:
            t2.append(l)
    if len(t2)+1!=len(t1):
        return None
    for i in range(2):
        if i+len(t2)>len(t1):
            return None
        if t2==t1[i:i+len(t2)]:
            return lf_1[:i]+['copy',(lf_1[i:i+len(t2)])]+lf_1[i+len(t2):] 

    return None    

def match_lf_all(lf_1,lf_2):
    cont=0
    for l in lf_2:
        if type(l)==str and l.startswith('A'):
            cont+=1
    if cont<=2:
        return None
    t1=lf_1.copy()
    t2=lf_2.copy()
    for i in range(len(t1)):
        if i+len(t2)>=len(t1):
            return None
        if t2==t1[i:i+len(t2)]:
            return lf_1[:i]+['copy_all',(lf_1[i:i+len(t2)])]+lf_1[i+len(t2):]       
    return None   

def parsing_lf(lf):
    res=[]
    cont=[]
    for l in lf:
        if l in action:
            nonterminal=-1
            temp=action[l][1]
            if type(temp)!=tuple:
                temp=(temp,)
            for s in temp:
                if s in state_action or s in ["r","e","Type","num"]:
                    nonterminal+=1
            for i in range(len(res)):
                if cont[i]!=0:
                    res[i]+=[l]
                    cont[i]+=nonterminal
            if type(action[l][1])==tuple:
                res.append([l])
                cont.append(nonterminal+1)
        else:
            for i in range(len(res)):
                if cont[i]!=0:
                    res[i]+=[l]
                    cont[i]-=1 
    return res
                
    
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess.py') 
    parser.add_argument('-num_each_type', type=int, default=15000,help="number of questions for each type")
    opt = parser.parse_args()
    #Get action and mask such that state to legal actions
    database=agent.KB(mode='online')
    parser=Parser.Parser(database)
    action=Parser.Base_action
    state_action=parser.build_state_action(action)
            
    #load dataset
    train=get_data()
    random.shuffle(train)
    hparams=create_hparams()
    
    #create training iterator
    train_iterator=data_iterator.TextIterator(hparams)
    
    #create copy subsequences dataset, paper A19-21
    it=0
    cont_dict={}
    data={}
    copy_all=0
    copy_e=0
    copy_p=0
    cont=0
    for it in tqdm(range(len(train)),total=len(train)):
        files=train[it:min(len(train),it+hparams['batch_size'])]
        it=min(len(train),it+hparams['batch_size'])
        try:
            train_iterator.reset(files)
        except:
            continue
            
        temp_lf=[]
        cont=0

        while True:
            try:
                #get the next turn question and context
                entities,pres,types,numbers,utterances,answers,question_type,lf=train_iterator.next()
                entities=entities[0]
                pres=pres[0]
                types=types[0]
                numbers=numbers[0]
                utterances=utterances[0]
                answers=answers[0]
                question_type=question_type[0]
                lf=lf[0]
            
                #introduce copy actions into action sequences of current question
                t_lf=lf.copy()
                cont+=1
                history_lf=[]
                if True:
                    for i in range(len(lf)):
                        find=False
                        for j in range(len(temp_lf)):
                            all_lf=parsing_lf(temp_lf[j])
                            for k in range(len(all_lf)):
                                l=all_lf[k]
                                res=match_lf_all(lf[i],l)
                                if res != None:
                                    lf[i]=(res,all_lf,k)
                                    copy_all+=1
                                    find=True
                                    break
                                res=match_lf_pred(lf[i],l)
                                if res != None:
                                    lf[i]=(res,all_lf,k)
                                    copy_p+=1
                                    find=True
                                    break
                                res=match_lf_entity(lf[i],l)
                                if res != None:
                                    lf[i]=(res,all_lf,k)
                                    copy_e+=1
                                    find=True
                                    break                       
                            if find:
                                break
                        if find is False:
                            lf[i]=(lf[i],[],-1)
            
                #for boolean type questions, only keep those question whose answer is "yes" as training data 
                if len(lf)>0 and (type(answers)!=list or answers[0]!='no'):
                    if question_type not in data:
                        data[question_type]=[]
                    valid=True
                    for p in pres:
                        for p_ in p:
                            if type(p_)!=str or p_.startswith('P') is False:
                                valid=False
                                break
                    if valid is False:
                        break
                    data[question_type].append((entities,pres,types,numbers,utterances,answers,question_type,lf))
                    if question_type not in cont_dict:
                        cont_dict[question_type]=0
                    cont_dict[question_type]+=1
                temp_lf=t_lf
            except StopIteration:
                break    
            
    #keep 15000 example for each question type        
    all_data=[]
    max_cont=opt.num_each_type
    for qt in data:
        data[qt]=random.sample(data[qt],min(max_cont,len(data[qt])))
        all_data.extend(data[qt])
    for qt in data:
        print(qt,len(data[qt]))
    print(len(all_data))
    pickle.dump(all_data,open('data/SMP/BFS_data.pkl','wb'))
    

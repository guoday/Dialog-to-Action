import BFS.parser as Parser
import sys
import os
import json
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pickle
import numpy
import random
from tqdm import tqdm
import BFS.agent as agent
import random
from SMP import data_iterator 
from SMP import model
import time
import signal
import torch
import timeout_decorator
from tqdm import tqdm
import argparse
def handler(signum, frame):
    raise AssertionError     

def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)

def create_hparams(opt):
    #build default hparams
    dic={}
    dic['batch_size']=1
    dic['encoder_vocab']=opt.encoder_vocab
    dic['decoder_vocab']=opt.decoder_vocab
    dic['hidden_size']=opt.hidden_size
    dic['depth']=opt.depth
    dic['learning_rate']=opt.lr
    dic['display']=opt.display
    dic['beamsize']=opt.beam_size
    dic['updatesize']=opt.batch_size
    dic['dev']=opt.dev_display
    dic['train_num']=opt.train_iter
    dic['test']=opt.test
    dic['mode']=opt.mode
    return dic


def build_vocabulary():
    #build decoder vocabulary
    vocab=['S','entity_gate_1','entity_gate_2','entity_gate_3','copy_all','copy','<pad>']
    for i in Parser.Base_action:
        vocab.append(i)
    with open('SMP/model/vocab.out','w') as f:
        for v in vocab:
            f.write(v+'\n')  

def get_data():
    #load dataset
    dir="data/BFS/"
    train=[]
    dev=[]
    test=[]
    for root, dirs, files in os.walk(dir):
        for file in files:
            temp=os.path.join(root,file)
            if '.json' in temp:
                if 'train' in temp:
                    train.append(temp)
                elif 'dev' in temp:
                    dev.append(temp)
                elif 'test' in temp:
                    test.append(temp)
    return train,dev,test
def get_example(data,it):
    entities,pres,types,numbers,utterances,answers,question_type,lf=data[it]
    #fliter all illegal relation and type
    temp=[]
    for p in pres:
        temp+=p
    pres=list(set(temp))
    temp=[]
    for p in pres:
        if p.startswith('P'):
            temp.append(p)
    pres=temp

    types=[]
    for p in pres:
        types+=type_pre_link_dic[p]
    types=list(set(types))
    temp=[]
    for t in types:
        if t.startswith('Q'):
            temp.append(t)
    types=temp
    lf=random.sample(lf,1)
    return entities,pres,types,numbers,utterances,answers,question_type,lf
def get_dev_example(iterator):
    entities,pres,types,numbers,utterances,answers,question_type,_=iterator.next()
    temp=[]
    for p in pres[0]:
        temp+=p
    pres=list(set(temp))
    temp=[]
    for p in pres:
        if p.startswith('P'):
            temp.append(p)
    pres=temp

    types=[]
    for p in pres:
            types+=type_pre_link_dic[p]
    types=list(set(types))
    temp=[]
    for t in types:
        if t.startswith('Q'):
            temp.append(t)
    types=temp      
    return entities,pres,types,numbers,utterances,answers,question_type,_  
def filters(entities,predicates):
    #if entity have no link to all realtions, filt them
    entites=list(set(entities))
    filters_e=[]
    for e in entites:
          for predicate in predicates:
            if database.entity_link_predicate(e,predicate):
                filters_e.append(e)
                break
    return filters_e  

def Eval(find,top1,pred_answer,recall,precision,top1_pred,dev_preds,dev_dict,
         dev_top_beam_size,preds_dev,answers,last_question_type,question_type):
    dev_preds.append(top1)
    preds_dev.append(find)
    if question_type[0]!='Clarification':
        if question_type[0] not in dev_dict:
            dev_dict[question_type[0]]=[0.0,0.0]
        if top1 is False:
            dev_dict[question_type[0]][0]+=1
        else:
            dev_dict[question_type[0]][1]+=1 

        if type(answers[0])==set:
            if question_type[0] not in recall:
                recall[question_type[0]]=[]
                precision[question_type[0]]=[]
            if type(pred_answer)==set:
                if len(answers[0])==0 or len(pred_answer)==0:
                    recall[question_type[0]].append(0.0)
                    precision[question_type[0]].append(0.0) 
                else:
                    recall[question_type[0]].append(len(answers[0]&pred_answer)/len(answers[0]))
                    precision[question_type[0]].append(len(answers[0]&pred_answer)/len(pred_answer))
            else:
                recall[question_type[0]].append(0.0)
                precision[question_type[0]].append(0.0)
       
    if last_question_type=='Clarification' and question_type[0]!='Clarification':
        if type(answers[0])==set:
            if last_question_type not in recall:
                recall[last_question_type]=[]
                precision[last_question_type]=[]
            if type(pred_answer)==set:
                if len(answers[0])==0 or len(pred_answer)==0:
                    recall[last_question_type].append(0.0)
                    precision[last_question_type].append(0.0)  
                else:
                    recall[last_question_type].append(len(answers[0]&pred_answer)/len(answers[0]))
                    precision[last_question_type].append(len(answers[0]&pred_answer)/len(pred_answer))
            else:
                recall[last_question_type].append(0.0)
                precision[last_question_type].append(0.0) 
                
def Display(recall,precision,top1_pred,dev_preds,dev_dict,dev_top_beam_size,preds_dev):  
    print("Number of examples:",len(preds_dev))
    r=[]
    R={}
    for temp in recall:
        r+=recall[temp]
        R[temp]=np.mean(recall[temp])*100.0
    p=[]
    P={}
    for temp in precision:
        p+=precision[temp]
        P[temp]=np.mean(precision[temp])*100.0
    keys=sorted(list(set(dev_dict)|set(R)))
    print("-"*100)
    print("%-35s %-15s %-15s"%("","Recall","Precision"))
    print("%-35s %-17.2f %-15.2f"%("Overall",np.mean(r)*100.0,np.mean(p)*100.0))
    for k in keys:
        if k in R:
            print("%-35s %-17.2f %-15.2f"%(k,R[k],P[k]))
    print("-"*100)    
    print("%-43s %-15s"%("","Accuracy"))
    for k in keys:
        if k not in R:
            print("%-44s %-15.2f"%(k,dev_dict[k][1]*100.0/sum(dev_dict[k])))        
    return (np.mean(r)+np.mean(p))/2.0*100.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BFS/train.py')   
    parser.add_argument('-mode', required=True,help="offline or online")  
    parser.add_argument('-encoder_vocab', help="encoder vocabulary")
    parser.add_argument('-decoder_vocab', help="encoder vocabulary")
    parser.add_argument('-hidden_size', type=int,default=300,help="hidden size of decoder and encoder")
    parser.add_argument('-lr', type=float, default=0.001,help="learning rate")
    parser.add_argument('-beam_size', type=int, default=3,help="beam size of BFS")
    parser.add_argument('-batch_size', type=int, default=32,help="batch size")
    parser.add_argument('-depth', type=int, default=30,help="length of logical form by inference")
    parser.add_argument('-display', type=int, default=100,help="number of loss result")
    parser.add_argument('-dev_display', type=int, default=1500,help="number of dev result")
    parser.add_argument('-train_iter', type=int, default=15000,help="number of iterator")
    parser.add_argument('-test', action="store_true",help="test or train")

    #build hparams
    opt = parser.parse_args()
    print(opt)
    hparams=create_hparams(opt)
    #build vocabulary
    build_vocabulary()
    #build kb and parser
    database=agent.KB(mode=hparams['mode'])    
    parser=Parser.Parser(database)
    train,dev,test=get_data()
    #build dev data iterator
    dev_iterator=data_iterator.TextIterator(hparams)
    test_iterator=data_iterator.TextIterator(hparams)
    #build D2A model
    Model=model.D2A(parser,hparams,Parser.Base_action) 
    print(Model.parameters)
    if hparams['test'] is True: 
        Model.load_state_dict(torch.load('SMP/model/model.pkl'))

    #load trainning data
    train_data=pickle.load(open('data/SMP/BFS_data.pkl','rb'))
    print('loading done')
    print('Num of training examples:',len(train_data))
    random.seed(2018)
    random.shuffle(train_data)
    type_pre_link_dic=pickle.load(open('data/BFS/type_predicate_link.pkl','rb'))
    start_time=time.time()

    global_step=0
    best_score=0
    it=0
    train_loss=[]
    while global_step<hparams['train_num']:
        if it>=len(train_data):
            it=0
        if hparams['test'] is False:
            #get one example of training data
            entities,pres,types,numbers,utterances,answers,question_type,lf=get_example(train_data,it)
            it+=1
            update=Model.train(entities,pres,types,numbers,utterances,lf)
        if hparams['test'] is False and update:
            #update model parameter
            loss=Model.update()
            train_loss.append(loss)
            #display loss
            if (global_step+1)%hparams['display']==0:
                print("Iteration: "+str(global_step+1),"ppl:"+str(round(np.exp(np.mean(train_loss)),2)),
                      'Time: '+str(round(time.time()-start_time,2)))
                start_time=time.time()
                train_loss=[]
                train_preds=[]
                first_cont=0
            #eval dev dataset
            if (global_step+1)%hparams['dev']==0:
                num=min(200,len(dev))
                recall={}
                precision={}
                top1_pred=[] 
                dev_preds=[]
                dev_dict={}
                dev_top_beam_size={}
                preds_dev=[]
                for i in tqdm(range(num),total=num):
                    dev_iterator.reset(dev[i:i+1])
                    last_question_type=""
                    h_lf=[] #history top1 logical form, for copy action
                    while True:
                        try:  
                            entities,pres,types,numbers,utterances,answers,question_type,_=get_dev_example(dev_iterator)
                            try:
                                find,top1,pred_answer,h_lf=Model.infer(entities[0],pres,types,numbers[0],
                                                                       utterances[0],answers[0],hparams['beamsize'],h_lf)
                                if len(h_lf)!=0:
                                    h_lf=h_lf[1:]
                            except timeout_decorator.TimeoutError:
                                find=False
                                top1=False
                                pred_answer=[]
                                h_lf=[]
                                pass
                            Eval(find,top1,pred_answer,recall,precision,top1_pred,dev_preds,
                                 dev_dict,dev_top_beam_size,preds_dev,answers,last_question_type,question_type) 
                            last_question_type=question_type[0]  
                        except StopIteration:
                            break
                score=Display(recall,precision,top1_pred,dev_preds,dev_dict,dev_top_beam_size,preds_dev)   
                if best_score<score:
                    bad=0
                    best_score=score
                    print("Best score",best_score)
                    print("Model save")
                    torch.save(Model.state_dict(), 'SMP/model/model.pkl')
                else:
                    bad+=1
                    print("Best score",best_score)
            global_step+=1
            
        #eval test dataset
        if hparams['test']:
            print("Display test result each 100 dialog")
            num=len(test)
            recall={}
            precision={}
            top1_pred=[] 
            dev_preds=[]
            dev_dict={}
            dev_top_beam_size={}
            preds_dev=[]
            for i in tqdm(range(num),total=num):
                test_iterator.reset(test[i:i+1])
                last_question_type=""
                h_lf=[]
                while True:
                    try:  
                        entities,pres,types,numbers,utterances,answers,question_type,_=get_dev_example(test_iterator)
                        try:
                            find,top1,pred_answer,h_lf=Model.infer(entities[0],pres,types,numbers[0],
                                                                   utterances[0],answers[0],hparams['beamsize'],h_lf)
                            if len(h_lf)!=0:
                                h_lf=h_lf[1:]
                        except timeout_decorator.TimeoutError:
                            find=False
                            top1=False
                            pred_answer=[]
                            h_lf=[]
                            pass
                        Eval(find,top1,pred_answer,recall,precision,top1_pred,dev_preds,
                             dev_dict,dev_top_beam_size,preds_dev,answers,last_question_type,question_type) 
                        last_question_type=question_type[0]  
                    except StopIteration:
                        break
                if (i+1)%100==0:
                    Display(recall,precision,top1_pred,dev_preds,dev_dict,dev_top_beam_size,preds_dev) 
            Display(recall,precision,top1_pred,dev_preds,dev_dict,dev_top_beam_size,preds_dev)   
            break
        


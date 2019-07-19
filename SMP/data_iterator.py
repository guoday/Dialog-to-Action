import json
from nltk.tokenize import WordPunctTokenizer
def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)

#define Dialog Memory
class Memory(object):
    def __init__(self):
        self.entities=[[],[]]
        self.pre=[[]]
        self.history=['empty_context','empty_context']
        
    def current_state(self,entities,pre):
        #current state: entities and relations
        #for action subsequences, see SMP/model.py
        self.entities.append(entities)
        self.pre.append(pre)
        return self.entities,self.pre
    
    def update(self,entities_user,entities_sys,pre,question,answer):
        #update memory
        self.entities=[entities_user,entities_sys]
        self.history=[question,answer]
        self.pre=[pre]
            
    def context(self,question):
        #context:last question | response | current question
        return ' | '.join(self.history+[question])
    
    def clear(self):
        #clear dialog memory
        self.entities=[[],[]]
        self.pre=[[]]
        self.history=['empty_context','empty_context']
            
class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self,hparams):
        self.batch_size=hparams['batch_size']
        self.memory=[Memory() for i in range(self.batch_size)]
        
        
    def reset(self,files):
        assert len(files)<=self.batch_size
        self.current_size=len(files)
        self.it=0
        for m in self.memory:
            m.clear()
        self.dialogs=[json.load(open(f,'r')) for f in files]
        self.length=[len(d)//2 for d in self.dialogs]
        
    def next(self):
        if self.it>=max(self.length):
            raise StopIteration
        entities=[]
        pres=[]
        types=[]
        numbers=[]
        answers=[]
        utterances=[]
        question_type=[]
        True_lf=[]
        for i in range(self.current_size):
            if self.it>=self.length[i]:
                continue
                
            dialog_question=self.dialogs[i][self.it*2]
            dialog_response=self.dialogs[i][self.it*2+1]
            memory=self.memory[i]
            
            #get answer for this turn
            a=self.parsing_answer(dialog_response['all_entities'],dialog_response['utterance'],
                                  dialog_question['question-type'])
            
            #get entities from entities linking, relation from relation classifier, current question
            n=[]
            for x in dialog_question['utterance'].split():
                try:
                    n.append(int(x))
                except:
                    continue     
            n=list(set(n))
            e,p=memory.current_state(dialog_question['entities_linking'],dialog_question['predicate_prediction'])
            u=memory.context(dialog_question['utterance'])
            

            
            #update memory
            memory.update(dialog_question['entities_linking'],dialog_response['entities_linking'],
                          dialog_question['predicate_prediction'],dialog_question['utterance'],
                          dialog_response['utterance'])
            
            #obtain logical forms from BFS
            try:
                lf=[[l[1] for l in lf_[1]] for lf_ in dialog_response["true_lf"]]
            except:
                lf=[]
            entities.append(e)
            pres.append(p)
            types.append([])
            numbers.append(n)
            answers.append(a)
            utterances.append(u)
            True_lf.append(lf)
            question_type.append(dialog_question['question-type'])
            
        self.it+=1
        return entities,pres,types,numbers,utterances,answers,question_type,True_lf

    def parsing_answer(self,all_entities,utterance,question_type):
        bool_answer=[]
        cont_answer=[]
        sentence=tokenize(utterance).split()
        for x in sentence:
            if x=='yes' or x=='no':
                bool_answer.append(x)
            try:
                cont_answer.append(int(x))
            except:
                continue
        if 'Bool' in question_type and len(bool_answer)!=0:
            return bool_answer
        if 'Count' in question_type and len(cont_answer)!=0:
            return cont_answer
        return set(all_entities)        
        
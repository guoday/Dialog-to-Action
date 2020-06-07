import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import os
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import BFS.parser as Parser
import json
import random
from torch.nn import init
import time
import timeout_decorator
import pickle
from SMP.layer import EncoderRNN,EncoderTemplate,EncoderLf,DecoderRNN
torch.manual_seed(1)
use_cuda = torch.cuda.is_available()

def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)


class D2A(nn.Module):
    def __init__(self,parser,hparams,action):
        super(D2A, self).__init__()
        self.hparams=hparams
        self.parser=parser
        #load encoder and decoder vocabulary
        self.encoder_index2word,self.encoder_word2index=self.read_vocab(hparams['encoder_vocab'])
        self.decoder_index2word,self.decoder_word2index=self.read_vocab(hparams['decoder_vocab'])
        #build encoder for utterence
        self.encoder=EncoderRNN(len(self.encoder_index2word),hparams['hidden_size']).cuda() 
        #build encoder for logical form with instantiation
        self.encoder_lf=EncoderLf(hparams['hidden_size']).cuda() 
        #build encoder for logical form without instantiation
        self.encoder_tl=EncoderTemplate(hparams['hidden_size']).cuda() 
        #dictionary such that key is state and value is list of legal actions
        self.state_action=parser.build_state_action(action)
        #build decoder to generate action sequences
        self.decoder=DecoderRNN(len(self.decoder_index2word),hparams['hidden_size']).cuda()
        #grammar
        self.action=action
        #load all entities name to build vocabulary for entities
        self.entities_dict=json.load(open('data/kb/items_wikidata_n.json'))
        #convert type to index, to look up embedding
        self.types2index={}
        #convert relation to index, to look up embedding
        self.pres2index={}   
        #linear layer for attention
        self.W_entities = nn.Linear(hparams['hidden_size'], hparams['hidden_size']*4).cuda()
        self.W_types = nn.Linear(hparams['hidden_size'], hparams['hidden_size']*4).cuda()
        self.W_numbers = nn.Linear(hparams['hidden_size'], hparams['hidden_size']*4).cuda()
        self.W_pres = nn.Linear(hparams['hidden_size'], hparams['hidden_size']*4).cuda()
        self.W_template = nn.Linear(hparams['hidden_size'], hparams['hidden_size']*4).cuda()
        self.W_lf = nn.Linear(hparams['hidden_size'], hparams['hidden_size']*4).cuda()
        #build types2index and pres2index
        self.cont=0
        self.loss=0
        for params in self.parameters():
            try:
                init.xavier_normal(params)
            except:
                nn.init.normal(params, mean=0, std=1)
        for w in self.encoder_word2index:
            try:
                int(w[1])
                if w.startswith('q'):
                    self.types2index[w]=self.encoder_word2index[w]
                if w.startswith('p'):
                    self.pres2index[w]=self.encoder_word2index[w]
            except:
                continue

        self.optimizer = optim.Adam(self.parameters(), lr=hparams['learning_rate'])
        
    def read_vocab(self,vocab):
        index2word={}
        word2index={}
        index=0
        with open(vocab,'r') as f:
            for word in f:
                index2word[index]=word.strip()
                word2index[word.strip()]=index
                index+=1
        return index2word,word2index

    def parsing_lf(self,lf):
        """
        args: 
            lf:a logical form
        return: 
            list:all subtrees
        """
        res=[]
        cont=[]
        for l in lf:
            if l in self.action:
                nonterminal=-1
                temp=self.action[l][1]
                if type(temp)!=tuple:
                    temp=(temp,)
                for s in temp:
                    if s in self.state_action or s in ["r","e","Type","num_utterence"]:
                        nonterminal+=1
                for i in range(len(res)):
                    if cont[i]!=0:
                        res[i]+=[l]
                        cont[i]+=nonterminal
                if type(self.action[l][1])==tuple:
                    res.append([l])
                    cont.append(nonterminal+1)
            else:
                for i in range(len(res)):
                    if cont[i]!=0:
                        res[i]+=[l]
                        cont[i]-=1 
        return res
    
    def look_up(self,word):
        """
        args:
            word: token
        return:
            embedding of token
        """
        if word in self.decoder_word2index:
            word_ids=Variable(torch.LongTensor([self.decoder_word2index[word]])).cuda()
            embed=self.decoder.embedding(word_ids).view(1,1,-1)
        elif word.lower() in self.pres2index:
            word_ids=Variable(torch.LongTensor([self.pres2index[word.lower()]])).cuda()
            embed=self.encoder.embedding(word_ids).view(1,1,-1)   
        elif word.lower() in self.types2index:
            word_ids=Variable(torch.LongTensor([self.types2index[word.lower()]])).cuda()
            embed=self.encoder.embedding(word_ids).view(1,1,-1)
        elif word.lower() in self.encoder_word2index:
            word_ids=Variable(torch.LongTensor([self.encoder_word2index[word.lower()]])).cuda()
            embed=self.encoder.embedding(word_ids).view(1,1,-1)                       
        elif word in self.entities_dict:
            word_ids=[self.encoder_word2index[w] if w in self.encoder_word2index else 0 for w in tokenize(self.entities_dict[word]).split()]
            word_ids=Variable(torch.LongTensor(word_ids)).cuda()
            embed=torch.mean(self.encoder.embedding(word_ids),0).view(1,1,-1)
        else:
            word_ids=[0]
            word_ids=Variable(torch.LongTensor(word_ids)).cuda()
            embed=torch.mean(self.encoder.embedding(word_ids),0).view(1,1,-1)
        return embed
    
    """Given all entities,relations,types,numbers and logical forms, return their embedding"""  
    def build_embedding(self,entities,pres,types,numbers,lf):
         ###relations embedding##
        if len(pres)==0:
            pres_embedding_ori=None
            pres_embedding=None
        else:
            pres_id=[self.pres2index[p.lower()] for p in pres]
            pres_ids=Variable(torch.LongTensor(pres_id)).cuda()
            pres_embedding_ori=self.encoder.embedding(pres_ids)
            pres_embedding=F.tanh(self.W_pres(pres_embedding_ori))
                        
        ###entities embedding###
        entities_embedding_ori=[]
        entities_embedding=[]
        for entity in entities:
            if len(entity)==0:
                entities_embedding_ori.append(None)
                entities_embedding.append(None)  
            else:
                entities_id=[[self.encoder_word2index[w] if w in self.encoder_word2index else 0 for w in tokenize(self.entities_dict[e]).split()] for e in entity]
                entities_ids=[Variable(torch.LongTensor(x)).cuda() for x in entities_id]
                entities_embedding_ori.append(torch.cat([ torch.mean(self.encoder.embedding(e),0).view(1,-1) for e in entities_ids],0))
                entities_embedding.append(F.tanh(self.W_entities(entities_embedding_ori[-1])))
                    
        ##types embedding##  
        if len(types)==0:
            types_embedding_ori=None
            types_embedding=None
        else:
            types_id=[self.types2index[t.lower()] for t in types]
            types_ids=Variable(torch.LongTensor(types_id)).cuda()
            types_embedding_ori=self.encoder.embedding(types_ids)
            types_embedding=F.tanh(self.W_types(types_embedding_ori))
            
        ##numbers embedding##    
        if len(numbers)==0:
            numbers_embedding_ori=None
            numbers_embedding=None
        else:
            numbers_id=[self.encoder_word2index[str(n)] if str(n) in self.encoder_word2index else 0 for n in numbers]
            numbers_ids=[Variable(torch.LongTensor([x])).cuda() for x in numbers_id]
            numbers_embedding_ori=torch.cat([ self.encoder.embedding(n).view(1,-1) for n in numbers_ids],0)
            numbers_embedding=F.tanh(self.W_numbers(numbers_embedding_ori))        
              
        copy_action=[]
        history_lf=lf    
        ##action subsequences without instantiation embedding##
        template=[]
        for t_l in history_lf:
            temp=[]
            copy_action.append(t_l[0])
            for l in t_l:
                if type(l)==str and l.startswith('A'):
                    temp.append(l)
            template.append(temp)
        if len(template)==0:
            template_embedding_ori=None
            template_embedding=None
        else:
            template_len=[len(l) for l in template]
            max_len=max(template_len)
            template=[l+['<pad>']*(max_len-len(l)) for l in template]
            inputs=[[self.decoder_word2index[x] for x in l] for l in template]
            encoder_hidden = self.encoder.initHidden().cuda()
            encoders_input = self.decoder.embedding(Variable(torch.LongTensor(inputs)).cuda().permute([1,0]))        
            encoder_hidden = self.encoder_tl.initHidden(len(inputs)).cuda()
            encoder_output,encoder_hidden= self.encoder_tl(encoders_input, encoder_hidden)
            template_embedding_ori=[]
            template_embedding=[]
            for i in range(len(template_len)):
                template_embedding_ori.append(encoder_output[template_len[i]-1,i])
                template_embedding.append(F.tanh(self.W_template(template_embedding_ori[-1])).view(1,-1)) 
            template_embedding=torch.cat(template_embedding,0)

        ##action subsequences with instantiation embedding##
        lfs=lf
        if len(lfs)==0:
            lf_embedding_ori=None
            lf_embedding=None
        else:
            lfs_len=[len(l) for l in lfs]
            max_len=max(lfs_len)
            lfs=[l+['<pad>']*(max_len-len(l)) for l in lfs]
            inputs=[]
            for lf_ in lfs:
                temp=[]
                for word in lf_:
                    word=str(word)
                    embed=self.look_up(word) 
                    temp.append(embed)
                inputs.append(torch.cat(temp,0))

            encoders_input=torch.cat(inputs,1)
            encoder_hidden = self.encoder_lf.initHidden(len(inputs)).cuda()
            encoder_output,encoder_hidden= self.encoder_lf(encoders_input, encoder_hidden)
            lf_embedding_ori=[]
            lf_embedding=[]
            for i in range(len(lfs_len)):
                lf_embedding_ori.append(encoder_output[lfs_len[i]-1,i])
                lf_embedding.append(F.tanh(self.W_lf(lf_embedding_ori[-1])).view(1,-1)) 
            lf_embedding=torch.cat(lf_embedding,0)
            
        return pres_embedding_ori,entities_embedding_ori,types_embedding_ori,numbers_embedding_ori,template_embedding_ori,lf_embedding_ori,pres_embedding,entities_embedding,types_embedding,numbers_embedding,template_embedding,lf_embedding
                
    

    """given decoder hidden, return all attention scores """
    def attention(self,hidden_att,pres_embedding,entities_embedding,
                  types_embedding,numbers_embedding,template_embedding,lf_embedding):       
        ###For relations,relations_softmax###
        if pres_embedding is None:
            pres_softmax=None
        else:
            pres_attention=torch.sum(hidden_att[:,None,:]*pres_embedding[None,:,:],-1)
            pres_softmax=F.softmax(pres_attention,-1)                            
        ###For entities,entities_softmax###
        entities_softmax=[]
        for entities_embed in entities_embedding:
            if entities_embed is None:
                entities_softmax.append(None)
            else:
                entities_attention=torch.sum(hidden_att[:,None,:]*entities_embed[None,:,:],-1)
                entities_softmax.append(F.softmax(entities_attention,-1))            
        ###For types,types_softmax####
        if types_embedding is None:
            types_softmax=None
        else:
            types_attention=torch.sum(hidden_att[:,None,:]*types_embedding[None,:,:],-1)
            types_softmax=F.softmax(types_attention,-1)            
        ###For numbers,numbers_softmax###
        if numbers_embedding is None:
            numbers_softmax=None
        else:
            numbers_attention=torch.sum(hidden_att[:,None,:]*numbers_embedding[None,:,:],-1)
            numbers_softmax=F.softmax(numbers_attention,-1)                  
        ###For template softmax###
        if template_embedding is None:
            template_softmax=None
        else:
            template_attention=torch.sum(hidden_att[:,None,:]*template_embedding[None,:,:],-1)
            template_softmax=F.softmax(template_attention,-1)                  
        ###For template softmax###
        if lf_embedding is None:
            lf_softmax=None
        else:
            lf_attention=torch.sum(hidden_att[:,None,:]*lf_embedding[None,:,:],-1)
            lf_softmax=F.softmax(lf_attention,-1)                
        return pres_softmax,entities_softmax,types_softmax,numbers_softmax,template_softmax,lf_softmax    
    
    def train(self,entities,pres,types,numbers,utterances,lf):
        pres_embedding_ori,entities_embedding_ori,types_embedding_ori,\
        numbers_embedding_ori,template_embedding_ori,lf_embedding_ori,\
        pres_embedding,entities_embedding,types_embedding,numbers_embedding,\
        template_embedding,lf_embedding=self.build_embedding(entities,pres,types,numbers,lf[0][1])

        #Encoding
        inputs=[self.encoder_word2index[x] if x in self.encoder_word2index else 0 for x in tokenize(utterances).split()+['<eos>']]
        encoder_hidden = self.encoder.initHidden().cuda()
        encoders_input = Variable(torch.LongTensor(inputs)).cuda()
        encoder_output,encoder_hidden= self.encoder(encoders_input, encoder_hidden)                          
        decoder_hidden=encoder_hidden.view(-1)

        #rebuild action sequences which has copy action
        copy_action=[]
        history_lf=lf[0][1]    
        ##action subsequences without instantiation embedding##
        template=[]
        for t_l in history_lf:
            temp=[]
            copy_action.append(t_l[0])
            for l in t_l:
                if type(l)==str and l.startswith('A'):
                    temp.append(l)
            template.append(temp)
            
        copy_template=False
        copy_all=False
        outputs=[lf[0][0]]
        loss=0
        for i in range(len(outputs)):
            output=outputs[i]
            temp_lf=[]
            copy_gate=[]
            for t in output:
                if type(t)==list:
                    for t_ in t:
                        if copy_all:
                            temp_lf.append(t_)
                            copy_gate.append(1)
                        elif copy_template:
                            if type(t_)==str and t_.startswith('A'):
                                temp_lf.append(t_)
                                copy_gate.append(1)
                            else:
                                temp_lf.append(t_)
                                copy_gate.append(1)
                elif t=="copy":
                    copy_template=True
                    temp_lf.append(t)
                    copy_gate.append(lf[0][2]) 
                elif t=="copy_all":
                    copy_all=True
                    temp_lf.append(t)
                    copy_gate.append(lf[0][2]) 
                else:
                    temp_lf.append(t)
                    copy_gate.append(1)   
                    
            #decoder
            output=temp_lf
            input=['S']+output[:-1]                    
            decoder_inputs=[]
            decoder_hidden_=decoder_hidden
            
            #get input embedding
            for step in range(len(input)):
                if input[step] in self.decoder_word2index:
                    word_ids=Variable(torch.LongTensor([self.decoder_word2index[input[step]]])).cuda()
                    embed=self.decoder.embedding(word_ids).view(1,1,-1)
                    decoder_inputs.append(embed)
                elif input[step] in entities[0]:
                    idx=entities[0].index(input[step])                     
                    decoder_inputs.append(entities_embedding_ori[0][idx].view(1,1,-1))
                elif input[step] in entities[1]:
                    idx=entities[1].index(input[step])                     
                    decoder_inputs.append(entities_embedding_ori[1][idx].view(1,1,-1))
                elif input[step] in entities[2]:
                    idx=entities[2].index(input[step])                     
                    decoder_inputs.append(entities_embedding_ori[2][idx].view(1,1,-1)) 
                elif input[step] in types:
                    idx=types.index(input[step])                    
                    decoder_inputs.append(types_embedding_ori[idx].view(1,1,-1)) 
                elif input[step] in numbers:
                    idx=numbers.index(input[step])                    
                    decoder_inputs.append(numbers_embedding_ori[idx].view(1,1,-1))   
                elif input[step] in pres:  
                    idx=pres.index(input[step])  
                    decoder_inputs.append(pres_embedding_ori[idx].view(1,1,-1)) 
                else:
                    return False 
                
            #decoding
            for step in range(len(output)):
                action_softmax, decoder_hidden_,hidden_att=self.decoder(decoder_inputs[step].view(1,1,-1), decoder_hidden_.view(1,1,-1),encoder_output) 
                
                #attention to all dialog memory
                pres_softmax,entities_softmax,types_softmax,numbers_softmax,template_softmax,lf_softmax=\
                self.attention(hidden_att,pres_embedding,entities_embedding,
                               types_embedding,numbers_embedding,template_embedding,lf_embedding)
                
                #calculate loss 
                if output[step] in self.decoder_word2index:
                    if output[step] in self.action:
                        action=self.state_action[self.action[output[step]][0]].copy()
                    else:
                        action=self.state_action[self.action[output[step+1]][0]].copy()
                    if len(set(copy_action)&set(action))!=0:
                        action.append('copy')
                        action.append('copy_all')  
                    action_score=action_softmax[0]
                    mask=[1.0 if self.decoder_index2word[i] in action else 0.0 for i in  range(len(self.decoder_index2word))]
                    mask=Variable(torch.FloatTensor(mask)).cuda()
                    action_score=(action_score+0.000001)*mask/(torch.sum((action_score+0.000001)*mask))
                    if output[step]=='copy':
                        loss+=-torch.log(action_score[self.decoder_word2index[output[step]]]+0.0001)
                        mask=[]
                        for a in copy_action:
                            if a in action:
                                mask.append(1.0)
                            else:
                                mask.append(0.0)
                        assert sum(mask)!=0
                        mask=Variable(torch.FloatTensor(mask)).cuda()
                        copy_score=template_softmax[0]
                        copy_score=(copy_score+0.000001)*mask/(torch.sum((copy_score+0.000001)*mask))
                        loss+=-torch.log(copy_score[copy_gate[step]]+0.0001)       
                    elif output[step]=='copy_all':
                        loss+=-torch.log(action_score[self.decoder_word2index[output[step]]]+0.0001)
                        mask=[]
                        for a in copy_action:
                            if a in action:
                                mask.append(1.0)
                            else:
                                mask.append(0.0)
                        assert sum(mask)!=0
                        mask=Variable(torch.FloatTensor(mask)).cuda()
                        copy_score=lf_softmax[0]
                        copy_score=(copy_score+0.000001)*mask/(torch.sum((copy_score+0.000001)*mask))
                        loss+=-torch.log(copy_score[copy_gate[step]]+0.0001)
                    elif copy_gate[step]==1:
                        loss+=-torch.log(action_score[self.decoder_word2index[output[step]]]+0.0001) 
                elif output[step] in entities[2]:
                    if copy_gate[step]==1:
                        action=['entity_gate_1','entity_gate_2','entity_gate_3']
                        action_score=action_softmax[0]
                        mask=[1.0 if self.decoder_index2word[i] in action else 0.0 for i in  range(len(self.decoder_index2word))]
                        mask=Variable(torch.FloatTensor(mask)).cuda()
                        action_score=(action_score+0.000001)*mask/(torch.sum((action_score+0.000001)*mask))
                        loss+=-torch.log(entities_softmax[2][0][entities[2].index(output[step])]+0.0001)
                        loss+=-torch.log(action_score[self.decoder_word2index['entity_gate_3']]+0.0001) 
                elif output[step] in entities[1]:
                    if copy_gate[step]==1:
                        action=['entity_gate_1','entity_gate_2','entity_gate_3']
                        action_score=action_softmax[0]
                        mask=[1.0 if self.decoder_index2word[i] in action else 0.0 for i in  range(len(self.decoder_index2word))]
                        mask=Variable(torch.FloatTensor(mask)).cuda()
                        action_score=(action_score+0.000001)*mask/(torch.sum((action_score+0.000001)*mask))
                        loss+=-torch.log(entities_softmax[1][0][entities[1].index(output[step])]+0.0001)
                        loss+=-torch.log(action_score[self.decoder_word2index['entity_gate_2']]+0.0001) 
                elif output[step] in entities[0]:
                    if copy_gate[step]==1:
                        action=['entity_gate_1','entity_gate_2','entity_gate_3']
                        action_score=action_softmax[0]
                        mask=[1.0 if self.decoder_index2word[i] in action else 0.0 for i in  range(len(self.decoder_index2word))]
                        mask=Variable(torch.FloatTensor(mask)).cuda()
                        action_score=(action_score+0.000001)*mask/(torch.sum((action_score+0.000001)*mask))                        
                        loss+=-torch.log(entities_softmax[0][0][entities[0].index(output[step])]+0.0001)
                        loss+=-torch.log(action_score[self.decoder_word2index['entity_gate_1']]+0.0001) 
                elif output[step] in types:  
                    if copy_gate[step]==1:       
                        loss+=-torch.log(types_softmax[0][types.index(output[step])]+0.0001)  
                elif output[step] in numbers:   
                    loss+=-torch.log(numbers_softmax[0][numbers.index(output[step])]+0.0001)   
                elif output[step] in pres:  
                    if copy_gate[step]==1:
                        loss+=-torch.log(pres_softmax[0][pres.index(output[step])]+0.0001)
                else:
                    return False 
                
        #keep loss until updatesize(beam size) times
        self.loss+=loss/len(outputs)
        self.cont+=1
        if self.cont>=self.hparams['updatesize']:
            return True
        else:
            return False
        
    def update(self):
        #update model parameter
        self.loss/=self.cont
        if np.isnan(float(self.loss)):
            print("Nan")
            self.cont=0
            self.loss=0 
            return 0                
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        l=float(self.loss)
        self.cont=0
        self.loss=0 
        return l   
        
    @timeout_decorator.timeout(10)
    def infer(self,entities,pres,types,numbers,utterances,answers,beam_size,history_lf):
        history_menory={}
        lfs=self.parsing_lf(history_lf)

        pres_embedding_ori,entities_embedding_ori,types_embedding_ori,\
        numbers_embedding_ori,template_embedding_ori,lf_embedding_ori,\
        pres_embedding,entities_embedding,types_embedding,numbers_embedding,\
        template_embedding,lf_embedding=self.build_embedding(entities,pres,types,numbers,lfs)

        copy_action=[]
        template=[]
        for t_l in lfs:
            temp=[]
            copy_action.append(t_l[0])
            for l in t_l:
                if type(l)==str and l.startswith('A'):
                    temp.append(l)
            template.append(temp)
            
        #Encoding
        inputs=[self.encoder_word2index[x] if x in self.encoder_word2index else 0 for x in tokenize(utterances).split()+['<eos>']]
        encoder_hidden = self.encoder.initHidden().cuda()
        encoders_input = Variable(torch.LongTensor(inputs)).cuda()
        encoder_output,encoder_hidden= self.encoder(encoders_input, encoder_hidden)                          
        decoder_hidden=encoder_hidden.view(-1)

        #Decoding
        right_stack=[['S']]
        right_depth=[[0]]
        left_stack=[[]]
        left_depth=[[]]
        action_seq=[['S']]
        copy_seq=[[]]
        left_state=[[]]
        score_hidden=[(Variable(torch.FloatTensor([0.0])).cuda(),decoder_hidden)]
        for _ in range(self.hparams['depth']):
            batch_size=len(action_seq)
            inputs=[]
            for i in range(batch_size):
                word=str(action_seq[i][-1])               
                embed=self.look_up(word)    
                inputs.append(embed)                    
            decoder_inputs=torch.cat(inputs,1)

            decoder_hidden=torch.cat([x[1].view(1,-1) for x in score_hidden],0).view(1,-1,self.hparams['hidden_size']*2)
            action_softmax, decoder_hidden,hidden_att=self.decoder(decoder_inputs, decoder_hidden,encoder_output)
            """
               decoder hidden attention all relation,entities,types,logical form 
            """
            pres_softmax,entities_softmax,types_softmax,numbers_softmax,template_softmax,lf_softmax=\
            self.attention(hidden_att,pres_embedding,entities_embedding,
                  types_embedding,numbers_embedding,template_embedding,lf_embedding)
                
                

            #build tree    
            new_right_stack=[]
            new_right_depth=[]
            new_left_stack=[]
            new_left_depth=[]
            new_action_seq=[]
            new_score_hidden=[]
            new_copy_seq=[]
            new_left_state=[]
            for idx in range(batch_size):
                if len(right_stack[idx])==0:
                    #if top of stack is empty, pass it.
                    new_action_seq.append(action_seq[idx])
                    new_score_hidden.append(score_hidden[idx])
                    new_right_stack.append(right_stack[idx]) 
                    new_right_depth.append(right_depth[idx])
                    new_left_stack.append(left_stack[idx])
                    new_left_depth.append(left_depth[idx])
                    new_copy_seq.append(copy_seq[idx])
                    new_left_state.append(left_state[idx])
                    continue
                elif right_stack[idx][0]=='r':
                    #if top of stack is relation r, instantiate it
                    if len(copy_seq[idx])!=0 and copy_seq[idx][0].startswith('P'):
                        #if it's in action subsequence copied, follow the the copied action subsequence
                        score=None
                        if copy_seq[idx][0] in pres:
                            p_idx=pres.index(copy_seq[idx][0])
                            score=torch.log(pres_softmax[idx][p_idx]+0.000001)
                        if score is None:
                            continue
                        right_stack[idx][0]=copy_seq[idx][0]
                        action_seq[idx].append(copy_seq[idx].pop(0))
                        new_action_seq.append(action_seq[idx])
                        new_score_hidden.append((score_hidden[idx][0]+score,decoder_hidden[idx]))        
                        new_right_stack.append(right_stack[idx]) 
                        new_right_depth.append(right_depth[idx])
                        new_left_stack.append(left_stack[idx])
                        new_left_depth.append(left_depth[idx])
                        new_copy_seq.append(copy_seq[idx])
                        new_left_state.append(left_state[idx]) 
                    else:   
                        #otherwise instantiate it from dialog memory
                        for p_idx in range(len(pres)):
                            p=pres[p_idx]
                            action_seq_temp=action_seq[idx].copy()
                            right_stack_temp=right_stack[idx].copy()
                            left_stack_temp=left_stack[idx].copy()
                            right_depth_temp=right_depth[idx].copy()
                            left_depth_temp=left_depth[idx].copy()
                            copy_seq_temp=copy_seq[idx].copy()
                            left_state_temp=left_state[idx].copy()
                            score_hidden_temp=score_hidden[idx]     
                            right_stack_temp[0]=p
                            action_seq_temp+=[p]
                            score=torch.log(pres_softmax[idx][p_idx]+0.000001)
                            score_hidden_temp=(score_hidden_temp[0]+score,decoder_hidden[idx])
                            new_action_seq.append(action_seq_temp)
                            new_score_hidden.append(score_hidden_temp)
                            new_right_stack.append(right_stack_temp) 
                            new_right_depth.append(right_depth_temp)
                            new_left_stack.append(left_stack_temp)
                            new_left_depth.append(left_depth_temp)
                            new_copy_seq.append(copy_seq_temp)
                            new_left_state.append(left_state_temp)
                elif right_stack[idx][0]=='e':
                    #if top of stack is an instantiated action for entity e, instantiate it
                    action=['entity_gate_1','entity_gate_2','entity_gate_3']
                    action_score=action_softmax[idx]
                    mask=[1.0 if self.decoder_index2word[i] in action else 0.0 for i in  range(len(self.decoder_index2word))]
                    mask=Variable(torch.FloatTensor(mask)).cuda()
                    action_score=(action_score+0.000001)*mask/(torch.sum((action_score+0.000001)*mask)) 
                    if len(copy_seq[idx])!=0 and copy_seq[idx][0].startswith('Q'):
                        #if it's in action subsequence copied, follow the the copied action subsequence
                        score=None
                        for i in range(len(entities)):
                            if copy_seq[idx][0] in entities[i]:
                                p_idx=entities[i].index(copy_seq[idx][0])
                                score_=torch.log(entities_softmax[i][idx][p_idx]+0.000001)
                                score_+=torch.log(action_score[self.decoder_word2index['entity_gate_'+str(i+1)]]+0.000001)
                                if score is None or float(score)<float(score_):
                                    score=score_
                        if score is None:
                            continue
                        right_stack[idx][0]=copy_seq[idx][0]
                        action_seq[idx].append(copy_seq[idx].pop(0))
                        new_action_seq.append(action_seq[idx])
                        new_score_hidden.append((score_hidden[idx][0],decoder_hidden[idx]))        
                        new_right_stack.append(right_stack[idx]) 
                        new_right_depth.append(right_depth[idx])
                        new_left_stack.append(left_stack[idx])
                        new_left_depth.append(left_depth[idx])
                        new_copy_seq.append(copy_seq[idx])
                        new_left_state.append(left_state[idx])  
                    else:  
                        #otherwise instantiate it from dialog memory
                        for i in range(len(entities)):
                            for e_idx in range(len(entities[i])):
                                e=entities[i][e_idx]
                                action_seq_temp=action_seq[idx].copy()
                                right_stack_temp=right_stack[idx].copy()
                                left_stack_temp=left_stack[idx].copy()
                                right_depth_temp=right_depth[idx].copy()
                                left_depth_temp=left_depth[idx].copy()
                                copy_seq_temp=copy_seq[idx].copy()
                                left_state_temp=left_state[idx].copy()
                                score_hidden_temp=score_hidden[idx]     
                                right_stack_temp[0]=e
                                action_seq_temp+=[e]
                                score=torch.log(entities_softmax[i][idx][e_idx]+0.000001)
                                score+=torch.log(action_score[self.decoder_word2index['entity_gate_'+str(i+1)]]+0.000001)
                                score_hidden_temp=(score_hidden_temp[0]+score,decoder_hidden[idx])
                                new_action_seq.append(action_seq_temp)
                                new_score_hidden.append(score_hidden_temp)
                                new_right_stack.append(right_stack_temp) 
                                new_right_depth.append(right_depth_temp)
                                new_left_stack.append(left_stack_temp)
                                new_left_depth.append(left_depth_temp)  
                                new_copy_seq.append(copy_seq_temp)
                                new_left_state.append(left_state_temp)
                elif right_stack[idx][0]=='Type':
                    #if top of stack is an instantiated action for type t, instantiate it
                    if len(copy_seq[idx])!=0 and copy_seq[idx][0].startswith('Q'):
                        #if it's in action subsequence copied, follow the the copied action subsequence
                        score=None
                        if copy_seq[idx][0] in types:
                            p_idx=types.index(copy_seq[idx][0])
                            score=torch.log(types_softmax[idx][p_idx]+0.000001)
                        if score is None:
                            continue
                        right_stack[idx][0]=copy_seq[idx][0]
                        action_seq[idx].append(copy_seq[idx].pop(0))
                        new_action_seq.append(action_seq[idx])
                        new_score_hidden.append((score_hidden[idx][0],decoder_hidden[idx]))        
                        new_right_stack.append(right_stack[idx]) 
                        new_right_depth.append(right_depth[idx])
                        new_left_stack.append(left_stack[idx])
                        new_left_depth.append(left_depth[idx])
                        new_copy_seq.append(copy_seq[idx])
                        new_left_state.append(left_state[idx])      
                    else:  
                        #otherwise instantiate it from dialog memory
                        for t_idx in range(len(types)):
                            t=types[t_idx]
                            action_seq_temp=action_seq[idx].copy()
                            right_stack_temp=right_stack[idx].copy()
                            left_stack_temp=left_stack[idx].copy()
                            right_depth_temp=right_depth[idx].copy()
                            left_depth_temp=left_depth[idx].copy()
                            copy_seq_temp=copy_seq[idx].copy()
                            left_state_temp=left_state[idx].copy()
                            score_hidden_temp=score_hidden[idx]     
                            right_stack_temp[0]=t
                            action_seq_temp+=[t]
                            score=torch.log(types_softmax[idx][t_idx]+0.000001)
                            score_hidden_temp=(score_hidden_temp[0]+score,decoder_hidden[idx])
                            new_action_seq.append(action_seq_temp)
                            new_score_hidden.append(score_hidden_temp)
                            new_right_stack.append(right_stack_temp) 
                            new_right_depth.append(right_depth_temp)
                            new_left_stack.append(left_stack_temp)
                            new_left_depth.append(left_depth_temp)
                            new_copy_seq.append(copy_seq_temp)
                            new_left_state.append(left_state_temp)
                elif right_stack[idx][0]=="num_utterence":
                    #if top of stack is an instantiated action for num, instantiate it
                    if len(copy_seq[idx])!=0 and type(copy_seq[idx][0])==int:
                        #if it's in action subsequence copied, follow the the copied action subsequence
                        if copy_seq[idx][0] not in numbers:
                            continue
                        n_idx=numbers.index(copy_seq[idx][0])
                        score=torch.log(numbers_softmax[idx][n_idx]+0.000001)
                        right_stack[idx][0]=copy_seq[idx][0]
                        action_seq[idx].append(copy_seq[idx].pop(0))
                        new_action_seq.append(action_seq[idx])
                        new_score_hidden.append((score_hidden[idx][0]+score,decoder_hidden[idx]))        
                        new_right_stack.append(right_stack[idx]) 
                        new_right_depth.append(right_depth[idx])
                        new_left_stack.append(left_stack[idx])
                        new_left_depth.append(left_depth[idx])
                        new_copy_seq.append(copy_seq[idx])
                        new_left_state.append(left_state[idx])
                    else:
                        #otherwise instantiate it from dialog memory
                        for n_idx in range(len(numbers)):
                            n=numbers[n_idx]
                            action_seq_temp=action_seq[idx].copy()
                            right_stack_temp=right_stack[idx].copy()
                            left_stack_temp=left_stack[idx].copy()
                            right_depth_temp=right_depth[idx].copy()
                            left_depth_temp=left_depth[idx].copy()
                            copy_seq_temp=copy_seq[idx].copy()
                            left_state_temp=left_state[idx].copy()
                            score_hidden_temp=score_hidden[idx]     
                            right_stack_temp[0]=n
                            action_seq_temp+=[n]
                            score_hidden_temp=(score_hidden_temp[0]+torch.log(numbers_softmax[idx][n_idx]+0.000001),decoder_hidden[idx])
                            new_action_seq.append(action_seq_temp)
                            new_score_hidden.append(score_hidden_temp)
                            new_right_stack.append(right_stack_temp) 
                            new_right_depth.append(right_depth_temp)
                            new_left_stack.append(left_stack_temp)
                            new_left_depth.append(left_depth_temp)  
                            new_copy_seq.append(copy_seq_temp)
                            new_left_state.append(left_state_temp)
                else:
                    #otherwise the top of stack is non-instantiated action
                    #judge whether copy action subsequence from history
                    action=self.state_action[right_stack[idx][0]]
                    copy_flag=False
                    if len(set(copy_action)&set(action))!=0:
                        copy_flag=True
                    last_action=None
                    for item in action_seq[idx]:
                        if type(item)==str and item[0]=='A':
                            last_action=item  
                    #create mask for illegal action            
                    action_score=action_softmax[idx]
                    mask=[1.0 if self.decoder_index2word[i] in action else 0.0 for i in  range(len(self.decoder_index2word))]
                    if copy_flag:
                        mask[self.decoder_word2index['copy']]=1.0
                        mask[self.decoder_word2index['copy_all']]=1.0
                    mask=Variable(torch.FloatTensor(mask)).cuda()
                    action_score=(action_score+0.000001)*mask/(torch.sum((action_score+0.000001)*mask))
                    if len(copy_seq[idx])!=0:
                        #if the action is in copied action subsequence, follow it
                        action_seq_temp=action_seq[idx].copy()
                        right_stack_temp=right_stack[idx].copy()
                        left_stack_temp=left_stack[idx].copy()
                        right_depth_temp=right_depth[idx].copy()
                        left_depth_temp=left_depth[idx].copy()
                        copy_seq_temp=copy_seq[idx].copy()
                        left_state_temp=left_state[idx].copy()
                        score_hidden_temp=score_hidden[idx] 
                        a=copy_seq_temp.pop(0)
                        score=torch.log(action_score[self.decoder_word2index[a]]+0.000001)
                        score_hidden_temp=(score_hidden_temp[0]+score,decoder_hidden[idx])
                        action_seq_temp+=[a]
                        next_state=self.action[a][1]
                        if type(next_state)!=tuple:
                            right_stack_temp[0]=next_state
                        else:
                            right_stack_temp=list(next_state)+right_stack_temp[1:]
                            right_depth_temp=[right_depth_temp[0]+1 for i in range(len(next_state))]+right_depth_temp[1:]   
                        new_action_seq.append(action_seq_temp)
                        new_score_hidden.append(score_hidden_temp)
                        new_right_stack.append(right_stack_temp) 
                        new_right_depth.append(right_depth_temp)
                        new_left_stack.append(left_stack_temp)
                        new_left_depth.append(left_depth_temp) 
                        new_copy_seq.append(copy_seq_temp) 
                        new_left_state.append(left_state_temp)
                    else:
                        #otherwise, enumerate all legal actions
                        for a in action:
                            if Parser.prune(action_seq[idx],a):
                                continue
                            action_seq_temp=action_seq[idx].copy()
                            right_stack_temp=right_stack[idx].copy()
                            left_stack_temp=left_stack[idx].copy()
                            right_depth_temp=right_depth[idx].copy()
                            left_depth_temp=left_depth[idx].copy()
                            copy_seq_temp=copy_seq[idx].copy()
                            left_state_temp=left_state[idx].copy()
                            score_hidden_temp=score_hidden[idx]   
                            score=torch.log(action_score[self.decoder_word2index[a]]+0.000001)
                            score_hidden_temp=(score_hidden_temp[0]+score,decoder_hidden[idx])
                            action_seq_temp+=[a]
                            next_state=self.action[a][1]
                            if type(next_state)!=tuple:
                                right_stack_temp[0]=next_state
                            else:
                                right_stack_temp=list(next_state)+right_stack_temp[1:]
                                right_depth_temp=[right_depth_temp[0]+1 for i in range(len(next_state))]+right_depth_temp[1:]   
                            new_action_seq.append(action_seq_temp)
                            new_score_hidden.append(score_hidden_temp)
                            new_right_stack.append(right_stack_temp) 
                            new_right_depth.append(right_depth_temp)
                            new_left_stack.append(left_stack_temp)
                            new_left_depth.append(left_depth_temp) 
                            new_copy_seq.append(copy_seq_temp)
                            new_left_state.append(left_state_temp)

                        #if there exit action subsequence that we can copy
                        if copy_flag:
                            
                            mask=[]
                            for a in copy_action:
                                if a in action:
                                    mask.append(1.0)
                                else:
                                    mask.append(0.0)
                            assert sum(mask)!=0
                            mask=Variable(torch.FloatTensor(mask)).cuda()
                            template_score=template_softmax[idx]
                            template_score=(template_score+0.000001)*mask/(torch.sum((template_score+0.000001)*mask))
                            lf_score=lf_softmax[idx]
                            lf_score=(lf_score+0.000001)*mask/(torch.sum((lf_score+0.000001)*mask))
                            for i in range(len(template)):
                                ##copy templates, i.g. action subsequence without instantiation##
                                t_lf=template[i]
                                if copy_action[i] not in action:
                                    continue
                                action_seq_temp=action_seq[idx].copy()
                                right_stack_temp=right_stack[idx].copy()
                                left_stack_temp=left_stack[idx].copy()
                                right_depth_temp=right_depth[idx].copy()
                                left_depth_temp=left_depth[idx].copy()
                                copy_seq_temp=copy_seq[idx].copy()
                                left_state_temp=left_state[idx].copy()
                                score_hidden_temp=score_hidden[idx] 
                                score=torch.log(action_score[self.decoder_word2index['copy']]+0.000001)
                                score+=torch.log(template_score[i]+0.000001)
                                copy_seq_temp=['copy']
                                for l in t_lf:
                                    if l!="<pad>":
                                        copy_seq_temp.append(l)                            
                                a=copy_seq_temp.pop(0)
                                score_hidden_temp=(score_hidden_temp[0]+score,decoder_hidden[idx])
                                action_seq_temp+=[a]  
                                new_action_seq.append(action_seq_temp)
                                new_score_hidden.append(score_hidden_temp)
                                new_right_stack.append(right_stack_temp) 
                                new_right_depth.append(right_depth_temp)
                                new_left_stack.append(left_stack_temp)
                                new_left_depth.append(left_depth_temp) 
                                new_copy_seq.append(copy_seq_temp)
                                new_left_state.append(left_state_temp)
                            for i in range(len(lfs)):
                                ##copy logical forms, i.g. action subsequence with instantiation##
                                t_lf=lfs[i]
                                if copy_action[i] not in action:
                                    continue
                                #only copy at least two depth subtree
                                cont=0
                                for l in t_lf:
                                    if type(l)==str and l.startswith('A'):
                                        cont+=1
                                if cont<=2:
                                    continue
                                action_seq_temp=action_seq[idx].copy()
                                right_stack_temp=right_stack[idx].copy()
                                left_stack_temp=left_stack[idx].copy()
                                right_depth_temp=right_depth[idx].copy()
                                left_depth_temp=left_depth[idx].copy()
                                copy_seq_temp=copy_seq[idx].copy()
                                left_state_temp=left_state[idx].copy()
                                score_hidden_temp=score_hidden[idx] 
                                score=torch.log(action_score[self.decoder_word2index['copy_all']]+0.000001)
                                score+=torch.log(lf_score[i]+0.000001)
                                copy_seq_temp=['copy_all']
                                for l in t_lf:
                                    if l!="<pad>":
                                        copy_seq_temp.append(l)                            
                                a=copy_seq_temp.pop(0)
                                score_hidden_temp=(score_hidden_temp[0]+score,decoder_hidden[idx])
                                action_seq_temp+=[a] 
                                new_action_seq.append(action_seq_temp)
                                new_score_hidden.append(score_hidden_temp)
                                new_right_stack.append(right_stack_temp) 
                                new_right_depth.append(right_depth_temp)
                                new_left_stack.append(left_stack_temp)
                                new_left_depth.append(left_depth_temp) 
                                new_copy_seq.append(copy_seq_temp)
                                new_left_state.append(left_state_temp)                            
            #sort by scores
            index=[]    
            if len(new_right_stack)!=0:
                all_score=torch.cat([x[0] for x in new_score_hidden])
                topv, topi = all_score.topk(len(new_right_stack))
                for i in range(len(new_right_stack)):
                    index.append(int(topi[i]))            
            right_stack=[new_right_stack[i] for i in index]
            right_depth=[new_right_depth[i] for i in index]
            left_stack=[new_left_stack[i] for i in index]
            left_depth=[new_left_depth[i] for i in index]
            action_seq=[new_action_seq[i] for i in index]   
            score_hidden=[new_score_hidden[i] for i in index]
            copy_seq=[new_copy_seq[i] for i in index]  
            left_state=[new_left_state[i] for i in index]
            
            #prune all illegal logical form, e.g. lead to empty answer
            valid_index=[]
            Not_need_induce=["r","e","Type","num_utterence"]
            for idx in range(len(right_stack)):
                if len(valid_index)>=beam_size:
                    break
                if len(right_stack[idx])==0: 
                        valid_index.append(idx)
                        continue
                flag=True
                while True:
                    if right_stack[idx][0] in Not_need_induce or right_stack[idx][0] in self.state_action: 
                        break
                    left_stack[idx].append(right_stack[idx].pop(0))
                    left_depth[idx].append(right_depth[idx].pop(0))  
                    left_state[idx].append(left_stack[idx][-1])
                    while (len(right_stack[idx])==0 and len(left_stack[idx])>1) or (len(right_depth[idx])!=0 and left_depth[idx][-1]>right_depth[idx][0]):
                        current_depth=left_depth[idx][-1]
                        current_s=[]    
                        current_state=[]
                        while len(left_depth[idx])>0 and left_depth[idx][-1]==current_depth:
                            current_s.insert(0,left_stack[idx].pop(-1))
                            current_state.insert(0,left_state[idx].pop(-1))
                            left_depth[idx].pop(-1)
                        
                            
                        if len(current_s)>=2 and current_s[-1]==current_s[-2]:
                            flag=False
                            break
                        if tuple(current_state) in history_menory:
                            local_answer=history_menory[tuple(current_state)]
                        elif len(current_s)<=4:
                            local_answer=self.parser.op(current_s[0],current_s[1:])
                            history_menory[tuple(current_state)]=local_answer

                        #print(local_answer)
                        if local_answer is None:
                            flag=False
                            break
                        left_stack[idx].append(local_answer)    
                        left_depth[idx].append(current_depth-1)
                        left_state[idx].append(tuple(current_state))
    
                    if flag is False:
                            break
                    if len(right_stack[idx])==0:
                        break
                    
                if flag:
                    valid_index.append(idx)
                    
            #keep only valid logical forms
            right_stack=[right_stack[i] for i in valid_index]
            right_depth=[right_depth[i] for i in valid_index]
            left_stack=[left_stack[i] for i in valid_index]
            left_depth=[left_depth[i] for i in valid_index]
            action_seq=[action_seq[i] for i in valid_index]   
            score_hidden=[score_hidden[i] for i in valid_index]
            copy_seq=[copy_seq[i] for i in valid_index]
            left_state=[left_state[i] for i in valid_index]
            if len(right_stack)==0:
                break

        #find the index of correct logical form 
        gold_index=[]        
        for i in range(len(left_stack)):
            if len(left_stack[i])!=1:
                continue
            temp_answer=left_stack[i][0]
            if type(temp_answer)==int:
                temp_answer=[temp_answer]
            if temp_answer==answers:
                gold_index.append(i)
        
        #obtain top1 logical forms
        top1_pred=False
        pred_answer=[]
        top1_lf=[]
        if len(score_hidden)>=1:
                all_score=torch.cat([x[0] for x in score_hidden])
                topv, top1_idx = all_score.topk(1)
                if int(top1_idx) in gold_index:
                    top1_pred=True
                pred_answer=left_stack[int(top1_idx)][0]
                if len(right_stack[int(top1_idx)])==0:
                    top1_lf=action_seq[int(top1_idx)]
                    temp=[]
                    for l in top1_lf:
                        if type(l)!=str or l.startswith('copy') is False:
                            temp.append(l)
                            
                    top1_lf=temp

        find=False
        if len(gold_index)>0:
            find=True
        return find,top1_pred,pred_answer,top1_lf
        
        



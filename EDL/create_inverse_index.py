"""Generate a dictionary, given a string as key, return ids of entities with highest score"""
import json
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from fuzzywuzzy import fuzz
from tqdm import tqdm
def tokenize(sentence):
    sentence=WordPunctTokenizer().tokenize(sentence.lower())
    return ' '.join(sentence)


if __name__ == "__main__":
    #This json contains a dict with keys as the ids of wikidata entities 
    #and values as their string labels.
    dic=json.load(open('data/kb/items_wikidata_n.json','r'))


    #create a dict, given a string as key, return ids of entities with highest score
    inverse_index={}
    for ids in tqdm(dic,total=len(dic)):
        temp=tokenize(dic[ids]).split()
        for i in range(len(temp)):
            for j in range(i+1,len(temp)+1):
                words=' '.join(temp[i:j])
                if j-i+1<=len(temp)-2:
                    continue
                score=fuzz.ratio(words,temp)
                if words not in inverse_index or inverse_index[words]['score']<score:
                    inverse_index[words]={}
                    inverse_index[words]['score']=score
                    inverse_index[words]['idxs']=[ids]
                elif inverse_index[words]['score']==score:
                    inverse_index[words]['idxs'].append(ids)


    json.dump(inverse_index,open('data/EDL/inverse_index.json','w'))
    

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess
import numpy as np
import tensorflow as tf




__all__ = ["evaluate"]


def evaluate(label_file,pred_file):
    recall=[]
    precision=[]
    f1=open(label_file,'r')
    f2=open(pred_file,'r')
    
    for line in zip(f1,f2):
        label=[]
        pred=[]

        label_s=line[0].strip().split()
        pred_s=line[1].strip().split()

        flag=False
        index=-1
        for i in range(len(label_s)):
            if label_s[i]=="B":
                flag=True
                index=i
            elif label_s[i]=="S":
                flag=False
                label.append((i,i))
            elif label_s[i]=="M":
                pass
            elif label_s[i]=="O":
                flag=False
            elif label_s[i]=="E":
                flag=False
                label.append((index,i))
                
        flag=False
        index=-1
        for i in range(len(pred_s)):
            if pred_s[i]=="B":
                flag=True
                index=i
            elif pred_s[i]=="S":
                flag=False
                pred.append((i,i))
            elif pred_s[i]=="M":
                pass
            elif pred_s[i]=="O":
                flag=False
            elif pred_s[i]=="E":
                flag=False
                pred.append((index,i)) 
        cont=0
        for w in label:
            if w in pred:
                cont+=1

        if len(label)==0 and len(pred)==0:
            continue
        else:
            if len(label)!=0:
                    recall.append(cont*100.0/len(label))
            if len(pred)!=0:
                    precision.append(cont*100.0/len(pred))
            else:
                    precision.append(0)


    return np.mean(recall),np.mean(precision)
            

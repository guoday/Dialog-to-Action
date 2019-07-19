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
    acc=[]
    f1=open(label_file,'r')
    f2=open(pred_file,'r')
    
    for line in zip(f1,f2):
        glod=line[0].strip().split(' ||| ')[0]
        pred=line[1].strip()
        acc.append(glod==pred)

    return np.mean(acc)
            

# Introduction

Implement of the paper "Dialog-to-Action: Conversational Question Answering Over a Large-Scale Knowledge Base". The pipeline of D2A on CSQA includes Entity Detection and Linking **(EDL)**, Relation Classifier **(RC)** , Generating weakly-supervised data using breath frist search **(BFS)** , Semantic Parser **(SMP)**. 

***After cleaning the code, reproduced results are shown as the below. There are some inconsisitent number compared with the result reported in the paper. If you want to reproduce numbers reported in the paper, you can send an e-mail to ask for first version.***
```shell
----------------------------------------------------------------------------------------------------
                                    Recall          Precision
Overall                             71.15             70.97
Clarification                       43.26             38.77
Comparative Reasoning (All)         34.76             47.66
Logical Reasoning (All)             67.00             72.55
Quantitative Reasoning (All)        37.97             43.95
Simple Question (Coreferenced)      68.92             67.50
Simple Question (Direct)            85.51             82.74
Simple Question (Ellipsis)          85.26             79.00
----------------------------------------------------------------------------------------------------
                                            Accuracy
Comparative Reasoning (Count) (All)          14.05
Quantitative Reasoning (Count) (All)         37.36
Verification (Boolean) (All)                 40.81

```

# Requirements

- python3

- TensorFlow>=1.4

- pytorch

- pip install timeout_decorator fuzzywuzzy tqdm flask

  

# Download dataset
   ``` shell
   cd data
   bash download.sh
   cd ..
   ```


Or you can download dataset from [website](https://amritasaha1812.github.io/CSQA/download/).

1. Download and unzip [dialog files](https://drive.google.com/file/d/1dgf-Qjvhfv-_EWoDjrTCAY5CwYCw-djt/view) (**CSQA_v9.zip** ) to the data folder

   ``` shell
   unzip data/CSQA_v9.zip -d data/
   mv data/CSQA_v9 data/CSQA
   ```

2. Download [wikidata](https://drive.google.com/drive/folders/1ITcgvp4vZo1Wlb66d_SnHvVmLKIqqYbR) and  move all wikidata jsons to "data/kb"

   ``` shell
   mkdir data/kb
   ```

   

# Entity Detection and Linking (EDL)

1. Preprocessing data

   ```shell
   mkdir data/EDL
   python EDL/create_inverse_index.py
   python EDL/preprocess.py 
   ```

2. Training the detection model

   ``` shell
   mkdir EDL/model
   export CUDA_VISIBLE_DEVICES=0 
   python3 -u EDL/main.py \
       --vocab=data/EDL/vocab.in  \
       --train_prefix=data/EDL/train \
       --dev_prefix=data/EDL/dev  \
       --test_prefix=data/EDL/test \
       --dropout=0.5 \
       --num_layer=2 \
       --batch_size=32 \
       --optimizer=sgd \
       --learning_rate=1 \
       --num_train_steps=50000 \
       --num_display_steps=2000 \
       --num_eval_steps=10000  \
       --infer_size=512
   ```

   

# Relation Classifier **(RC)**

1. Preprocessing data

   ```shell
   mkdir data/RC
   python RC/prepocess.py
   ```

2. Training the classifier

   ```shell
   mkdir RC/model
   export CUDA_VISIBLE_DEVICES=0 
   python -u RC/main.py \
       --vocab_pre=data/RC/vocab  \
       --train_prefix=data/RC/train \
       --dev_prefix=data/RC/dev  \
       --test_prefix=data/RC/test \
       --dropout=0.5 \
       --num_layer=2 \
       --batch_size=32 \
       --optimizer=sgd \
       --learning_rate=1 \
       --num_train_steps=100000 \
       --num_display_steps=1000 \
       --num_eval_steps=10000  \
       --infer_size=512
   ```

# Weakly-supervised data (BFS)

1. Preprocess data

   We need to build knowledge and using **EDL**(**RC**) to find entities(relations) for BFS search

   ```shell
   mkdir data/BFS/ data/BFS/train data/BFS/dev data/BFS/test
   export CUDA_VISIBLE_DEVICES=0
   python BFS/preprocess.py
   ```

2. Generating logical forms using BFS

   There's two modes to generate logical forms (**offline** and **online**)
  
   - **offline**

     The offline mode can search in parallel and is faster than online mode. However, it's difficult to debug your BFS program, since you have to spend too much time to load knowledge base before searching. Therefore, I suggest you to use this mode if you ensure there's no problem in your BFS program.

      ```shell
     #In our experiment, we set max_train as 60k and beam size as 1000. 
     #If your resources can support it, you can use more and set larger beam size. 
     #Suggest that you set the number of parallel as large as possible.
     python BFS/run.py -mode offline -num_parallel 5 -beam_size 1000 -max_train 10000
      ```
     ***Note: max_train = 60k needs three days using 10 threads (one thread needs 70G~ memory). However, you don't have to wait for finishing this stage. For example, if you search 1% training data, you can leave this BFS program to run in the backend and jump to next stage to train D2A model.**
     
     ***If your resource is limited, you can send an e-mail to Daya Guo (guody5@mail2.sysu.edu.cn) and ask for the searched data.**
   - **online**

     The online mode can debug your BFS program easily. The idea is to load knowledge base as a server first, and then the BFS program access the server using HTTP. Therefore, you can start your BFS program quickly without loading the knowledge base, which can help you debug. However, it's too slow to generating logical forms using HTTP. I suggest you to use this mode only if you want to debug your BFS program.

     ```shell
     python BFS/server.py #using another terminal to run it
     python BFS/run.py -mode online -num_parallel 1  -beam_size 1000 -max_train 10000 
     ```
     
3. Oracle score of BFS

    | Question Type | Oracle Score |
    | :------- | :---------: |
    | Simple Question (Direct) | 96.3 |
    | Simple Question (Coreferenced) | 91.4 | 
    | Simple Question (Ellipsis) | 95.1 | 
    | Logical Reasoning (All) | 48.3 |
    | Quantitative Reasoning (All) | 42.4 |
    | Comparative Reasoning (All) | 25.5 |
    | Clarification | 0.9 |
    | Comparative Reasoning (Count) (All) | 33.2 |
    | Quantitative Reasoning (Count) (All) | 68.1 |
    | Verification (Boolean) | 78.3 |


# Semantic Parser (SMP)

1. Preprocessing data

    If your resources can support it, you can use more. 

   ```shell
   mkdir data/SMP
   python SMP/preprocess.py -num_each_type 15000
   ```

2. Training the D2A model

   There's two modes to run D2A model (**offline** and **online**), because inference is too slow, we only use 200 dialogs (about 2000~ examples) to eval. We use all test dataset to test our model, which needs about two days, so we will display result every 100 dialogs (about 10~ minutes). 

   - **offline**

     too slow to start the D2A model, but it's fast to inference

     ```shell
     export CUDA_VISIBLE_DEVICES=0 
     python3 SMP/train.py\
         -mode offline \
         -encoder_vocab SMP/model/vocab.in \
         -decoder_vocab SMP/model/vocab.out  \
         -hidden_size 300 \
         -lr 0.001 \
         -beam_size 3 \
         -batch_size 32 \
         -depth 30 \
         -display 100 \
         -dev_display 1500 \
         -train_iter 15000 
     ```

   - **online**

     quickly start to run the D2A model, but it's too slow to inference. only suit to debug

     ```shell
     python BFS/server.py #using another terminal to run it
     export CUDA_VISIBLE_DEVICES=0 
     python3 SMP/train.py\
         -mode online \
         -encoder_vocab SMP/model/vocab.in \
         -decoder_vocab SMP/model/vocab.out  \
         -hidden_size 300 \
         -lr 0.001 \
         -beam_size 3 \
         -batch_size 32 \
         -depth 30 \
         -display 100 \
         -dev_display 1500 \
         -train_iter 15000 
     ```

3. Testing the D2A model

   ```shell
   export CUDA_VISIBLE_DEVICES=0 
   python3 SMP/train.py\
       -mode offline \
       -encoder_vocab SMP/model/vocab.in \
       -decoder_vocab SMP/model/vocab.out  \
       -hidden_size 300 \
       -lr 0.001 \
       -beam_size 3 \
       -batch_size 32 \
       -depth 30 \
       -display 100 \
       -dev_display 1500 \
       -train_iter 15000 \
       -test
   ```

   

   

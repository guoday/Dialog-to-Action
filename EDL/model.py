import tensorflow as tf
from EDL.utils import misc_utils as utils
from EDL.utils import evaluation_utils
import EDL.data_iterator as data_iterator
from EDL.data_iterator import TextIterator
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
import time
import os
symbol={0:'O',1:'B',2:'M',3:'E',4:'S'}
def count_line(files):
    f=open(files)
    return len(f.readlines())

def print_step_info(prefix, global_step, info):
  """Print all info at the current global step."""
  utils.print_out(
      "%sstep %d lr %g step-time %.2fs ppl %.2f gN %.2f, %s" %
      (prefix, global_step, info["learning_rate"], info["avg_step_time"],
       info["train_ppl"], info["avg_grad_norm"], time.ctime())) 
    
class Model(object):
    def __init__(self,hparams,batch_size,mode):
        self.batch_size=batch_size
        self.source = tf.placeholder(shape=(None,None), dtype=tf.string)
        self.target = tf.placeholder(shape=(None,None), dtype=tf.int32)
        self.length=tf.placeholder(shape=(None,), dtype=tf.int32)
        self.vocab_table=lookup_ops.index_table_from_file(hparams.vocab, default_value=0)  
        self.mode=mode
        
        #build graph
        self.build_graph(hparams) 
        
        self.optimizer(hparams)
        self.saver = tf.train.Saver(tf.global_variables())
        self.answer=tf.transpose(tf.argmax(self.logits,-1))
        self.cont=tf.reduce_sum(tf.minimum(self.length,1) )
        params = tf.trainable_variables()
        if self.mode==tf.contrib.learn.ModeKeys.TRAIN:
            utils.print_out("# Trainable variables")
            for param in params:
                  utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                            param.op.device))
    def build_graph(self, hparams):
        encoder_outputs, encoder_state = self.build_encoder(hparams)
        self.logits=self.build_mlp(hparams,encoder_outputs)
        self.cost=self.compute_loss(hparams,self.logits)
        
    def build_encoder(self,hparams):
        with tf.variable_scope("encoder") as scope:
            input_id=self.vocab_table.lookup(self.source)
            vocab_size=count_line(hparams.vocab)
            emb=tf.get_variable('embedding',[vocab_size,300])
            input_id=tf.transpose(input_id)
            encoder_emb_inp = tf.nn.embedding_lookup(emb, input_id)
            fw_cell,bw_cell= self._build_encoder_cell(hparams)
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,encoder_emb_inp,dtype=tf.float32,
                                                                   sequence_length=self.length,time_major=True,swap_memory=True)

            return tf.concat(bi_outputs, -1), bi_state
        
    def  build_mlp(self,hparams,encoder_outputs):
        with tf.variable_scope("MLP") as scope:
            hidden_layer = layers_core.Dense(hparams.hidden_size,activation=tf.nn.tanh, use_bias=True, name="hidden_projection")
            hidden_outputs=hidden_layer(encoder_outputs)
            if hparams.dropout > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                hidden_outputs=tf.nn.dropout(hidden_outputs, 1.0-hparams.dropout)
            output_layer = layers_core.Dense(5, use_bias=False, name="output_projection") 
            logits=output_layer(hidden_outputs)
            return logits
        
    def compute_loss(self,hparams,logits):
        target_output = tf.transpose(self.target)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(self.length, tf.shape(crossent)[0], dtype=logits.dtype)
        target_weights = tf.transpose(target_weights)
        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(tf.reduce_sum(tf.minimum(self.length,1) ))
        return loss
        
    def _build_encoder_cell(self,hparams):
        num_bi_layers = int(hparams.num_layer / 2) 
        fw_cell_list=[]
        bw_cell_list=[]
        for i in range(num_bi_layers):
            single_cell = tf.contrib.rnn.BasicLSTMCell(hparams.num_units,forget_bias=hparams.forget_bias)
            if hparams.dropout > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - hparams.dropout))
                utils.print_out("  %s, dropout=%g " %(type(single_cell).__name__, hparams.dropout),new_line=False)      
            fw_cell_list.append(single_cell)
            single_cell = tf.contrib.rnn.BasicLSTMCell(hparams.num_units,forget_bias=hparams.forget_bias)
            if hparams.dropout > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - hparams.dropout))
                utils.print_out("  %s, dropout=%g " %(type(single_cell).__name__, hparams.dropout),new_line=False)      
            bw_cell_list.append(single_cell)

        if num_bi_layers == 1:  # Single layer.
            fw_cell=fw_cell_list[0]
            bw_cell=bw_cell_list[0]
        else:  # Multi layers
            fw_cell=tf.contrib.rnn.MultiRNNCell(fw_cell_list)
            bw_cell=tf.contrib.rnn.MultiRNNCell(bw_cell_list)
        return fw_cell,bw_cell
                
                
    def optimizer(self,hparams):
        if hparams.optimizer == "sgd":
            opt = tf.train.GradientDescentOptimizer(hparams.learning_rate)
        elif hparams.optimizer == "adam":
            opt = tf.train.AdamOptimizer(hparams.learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.cost,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  
        self.grad_norm =gradient_norm 
        self.update = opt.apply_gradients(zip(clipped_grads, params))
        
    def train(self,sess,iterator):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        holder=[self.source,self.target,self.length]
        return sess.run([self.update,self.cost,self.grad_norm],feed_dict=dict(zip(holder,iterator.next())))
    
    def decode(self,sess,iterator):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        holder=[self.source,self.target,self.length]
        return sess.run([self.answer,self.cont,self.length],feed_dict=dict(zip(holder,iterator.next())))
    
    def infer(self,sess,data):
        holder=[self.source,self.target,self.length]
        return sess.run(self.answer,feed_dict=dict(zip(holder,data)))
    
  
    
def infer(hparams,path):
    infer_graph = tf.Graph()
    with infer_graph.as_default():  
        config_proto = tf.ConfigProto(
        log_device_placement=0,
        allow_soft_placement=0)
        config_proto.gpu_options.allow_growth = True
        infer_model=Model(hparams,hparams.infer_size,tf.contrib.learn.ModeKeys.INFER)
        infer_sess=tf.Session(graph=infer_graph,config=config_proto)
        infer_sess.run(tf.global_variables_initializer())
        infer_sess.run(tf.tables_initializer())  
        saver = infer_model.saver
        saver.restore(infer_sess,os.path.join(path, "model"))
        return infer_sess,infer_model
        
def train(hparams):
    utils.save_hparams('EDL/model',hparams)
    #create data iterator
    train_iterator= TextIterator(hparams.train_src,hparams.train_tgt,hparams,hparams.batch_size,shuffle=True)
    dev_iterator= TextIterator(hparams.dev_src,hparams.dev_tgt,hparams,hparams.infer_size,shuffle=False)
    
    #build model
    train_graph = tf.Graph()
    infer_graph = tf.Graph()
    config_proto = tf.ConfigProto(
    log_device_placement=0,
    allow_soft_placement=0)
    config_proto.gpu_options.allow_growth = True
    
    with train_graph.as_default():
        train_model=Model(hparams,hparams.batch_size,tf.contrib.learn.ModeKeys.TRAIN)
        train_sess=tf.Session(graph=train_graph,config=config_proto)
        train_sess.run(tf.global_variables_initializer())
        train_sess.run(tf.tables_initializer())
        
    with infer_graph.as_default():    
        infer_model=Model(hparams,hparams.infer_size,tf.contrib.learn.ModeKeys.INFER)
        infer_sess=tf.Session(graph=infer_graph,config=config_proto)
        infer_sess.run(tf.global_variables_initializer())
        infer_sess.run(tf.tables_initializer())
        
    #train model
    global_step=0 
    train_loss=0
    train_norm=0
    while global_step < hparams.num_train_steps:
        start_time = time.time()
        try:
            _,cost,norm=train_model.train(train_sess,train_iterator)
        except StopIteration:
            continue
        global_step+=1
        train_loss+=cost
        train_norm+=norm
        if global_step%hparams.num_display_steps==0:
              info={}
              info['learning_rate']=hparams.learning_rate
              info["avg_step_time"]=(time.time()-start_time)/hparams.num_display_steps
              start_time = time.time()
              info["train_ppl"]= utils.safe_exp(train_loss / hparams.num_display_steps)
              info["avg_grad_norm"]=train_norm/hparams.num_display_steps
              train_loss=0
              train_norm=0
              print_step_info("  ", global_step, info)
        if global_step%hparams.num_eval_steps==0:
              saver = train_model.saver
              saver.save(train_sess,'EDL/model/model')
              with infer_graph.as_default(): 
                  saver.restore(infer_sess,'EDL/model/model')
                  with open('EDL/model/out_dev','w') as f:
                        dev_iterator.reset()
                        while True:
                            try:
                                pred,length,sequence_length=infer_model.decode(infer_sess,dev_iterator)
                                for i in range(length):
                                    p=[symbol[idx] for idx in pred[i]][:sequence_length[i]]
                                    f.write(' '.join(p)+'\n')
                            except StopIteration:
                                break
                  dev_recall,dev_precision=evaluation_utils.evaluate(hparams.dev_tgt,'EDL/model/out_dev')
                  print("recall and precision",dev_recall,dev_precision)

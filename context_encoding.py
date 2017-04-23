import numpy as np
import tensorflow as tf
import os
import logging
import load_data
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib import legacy_seq2seq
from question_encoding import *

class context_encoding(object):
    def __init__(self,config):
        self.c_bp_lstm=context_bottom_up_lstm(config)
        self.inputs=self.c_bp_lstm.sentences_root_states
        self.inputs=tf.expand_dims(self.inputs, 0) #[1 , sentence_num, hidden_dim]
        self.sentence_num=tf.gather(tf.shape(self.inputs),1)
        self.sentence_num_batch=tf.expand_dims(self.sentence_num, 0)  #[1]   
        with tf.variable_scope('context_lstm_forward'): 
            self.fwcell=rnn.BasicLSTMCell(config.hidden_dim, activation=tf.nn.tanh)
        with tf.variable_scope('context_lstm_backward'): 
            self.bwcell=rnn.BasicLSTMCell(config.hidden_dim, activation=tf.nn.tanh)
        with tf.variable_scope('context_bidirectional_chain_lstm'):
            self._fw_initial_state=self.fwcell.zero_state(1,dtype=tf.float32)
            self._bw_initial_state=self.bwcell.zero_state(1,dtype=tf.float32)
            chain_outputs, chain_state=tf.nn.bidirectional_dynamic_rnn(self.fwcell, self.bwcell, self.inputs, self.sentence_num_batch, initial_state_fw=self._fw_initial_state, initial_state_bw=self._bw_initial_state)

        chain_outputs=tf.concat(chain_outputs, 2) #[1, sentence_num, 2*hidden_dim]
        chain_outputs=tf.gather(chain_outputs, 0) #[sentence_num, 2*hidden_dim]
        
        self.c_td_lstm=context_top_down_lstm(config, self.c_bp_lstm, chain_outputs)
        self.sentences_final_states=self.get_tree_states(self.c_bp_lstm.sentences_hidden_states, self.c_td_lstm.sentences_hidden_states)
            
    def get_tree_states(self, sentences_bp_states,sentences_td_states):
        rev_td_states=tf.reverse(sentences_td_states, axis=[1])
        states=tf.concat(values=[sentences_bp_states, rev_td_states],axis=2)
        return states
class context_bottom_up_lstm(object):
    def __init__(self,config):
        self.max_sentence_num=20
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.num_emb = config.num_emb
        self.config=config
        self.sentence_num=None
        self.reg=self.config.reg  #regulizer parameter
        self.degree=config.degree  #  2, the N-ary
        self.add_placeholders()
        #batch_size * maxnodesize * emb_dim
        emb_leaves = self.add_embedding()
        self.add_model_variables()
        self.sentences_hidden_states = self.compute_sentences_states(emb_leaves)
        #[sentences, node_size,hidden_value]
        self.sentences_root_states=self.get_sentences_root_states(self.sentences_hidden_states)
        #[sentences_num, hidden_value]
    def get_sentences_root_states(self, sentences_states):
        def _get_root_states(x):
            states=tf.gather(x, tf.subtract(tf.gather(tf.shape(x),0),1))
            return states
        hidden_states = tf.map_fn(_get_root_states,sentences_states)
        return hidden_states

    def add_placeholders(self):       
        dim2=self.config.maxnodesize #parse tree node的数量
        #dim1=self.max_sentence_num  # max sentence num, parallel computing
        self.sentence_num=tf.placeholder(tf.int32,name='context_sentence_num')
        self.input = tf.placeholder(tf.int32,[None,dim2],name='context_input')
        self.input=tf.gather(self.input, tf.range(self.sentence_num))
        self.treestr = tf.placeholder(tf.int32,[None,dim2,2],name='context_tree')
        self.treestr = tf.gather(self.treestr, tf.range(self.sentence_num)) 
        self.dropout = tf.placeholder(tf.float32,name='context_dropout')
        self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.treestr,-1)),[1,2])
        self.n_inodes = self.n_inodes//2
        self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input,-1)),[1])

    def add_embedding(self):
        with tf.variable_scope("Embed",regularizer=None,reuse=True):
            #embedding=tf.get_variable('embedding',[self.num_emb,self.emb_dim],initializer=self.emb_mat, trainable=False)
            embedding=tf.get_variable('embedding')
            ix=tf.to_int32(tf.not_equal(self.input,-1))*self.input
            emb_tree=tf.nn.embedding_lookup(embedding,ix)
            #emb_tree [sentence_num, maxnodesize, emb_dim] 
            emb_tree=emb_tree*(tf.expand_dims(
                        tf.to_float(tf.not_equal(self.input,-1)),2))
            return emb_tree
    def calc_wt_init(self,fan_in=300):
        eps=1.0/np.sqrt(fan_in)
        return eps
    def add_model_variables(self):

        with tf.variable_scope("context_btp_Composition",
                                initializer=
                                tf.contrib.layers.xavier_initializer(),
                                regularizer=
                                tf.contrib.layers.l2_regularizer(self.config.reg
            )):

            cU = tf.get_variable("cU",[self.emb_dim,2*self.hidden_dim],initializer=tf.random_uniform_initializer(-self.calc_wt_init(),self.calc_wt_init()))
            cW = tf.get_variable("cW",[self.degree*self.hidden_dim,(self.degree+3)*self.hidden_dim],initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim)))
            cb = tf.get_variable("cb",[4*self.hidden_dim],initializer=tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))

    def process_leafs(self,emb):
        #emb: [num_leaves, emd_dim]    
        with tf.variable_scope("btp_Composition",reuse=True):
            cU = tf.get_variable("cU",[self.emb_dim,2*self.hidden_dim])
            cb = tf.get_variable("cb",[4*self.hidden_dim])
            b = tf.slice(cb,[0],[2*self.hidden_dim])
            #叶子节点没有input gate和forget gate,需要计算output gate 和Input value
            def _recurseleaf(x):
                #[1, emb_dim], [emb_dim, 2*self.hidden_dim]
                concat_uo = tf.matmul(tf.expand_dims(x,0),cU) + b
                #把concat_uo切割成
                #[1*hidden_dim] [1*hidden_dim]
                u,o = tf.split(axis=1,num_or_size_splits=2,value=concat_uo)
                o=tf.nn.sigmoid(o)
                u=tf.nn.tanh(u)
                c = u#tf.squeeze(u)
                h = o * tf.nn.tanh(c)
                hc = tf.concat(axis=1,values=[h,c])
                hc=tf.squeeze(hc)
                return hc
        hc = tf.map_fn(_recurseleaf,emb)
        #hc [num_leaves, 2*hidden_dim]
        return hc
    def compute_sentences_states(self,emb_batch):
        states_h=self.compute_states(emb_batch,0)
        #[1 nodenum hidden_dim]
        idx_batch=tf.constant(1)
        def _computestates(states, emb_batch, idx_batch):
            cur_states=self.compute_states(emb_batch,idx_batch)
            #[1* node_num ,hidden_value]
            states=tf.concat([states, cur_states], axis=0)
            idx_batch=tf.add(idx_batch,1)
            return states,emb_batch,idx_batch
        loop_cond=lambda a1,b1,idx_var: tf.less(idx_var, self.sentence_num)
        loop_vars=[states_h,emb_batch,idx_batch]
        states_h,emb_batch,idx_batch=tf.while_loop(loop_cond, _computestates, loop_vars, 
            shape_invariants=[tf.TensorShape([None,None,self.hidden_dim]),emb_batch.get_shape(),idx_batch.get_shape()])
        return states_h  #[sentence_num, node_size, hidden_dim]
    def compute_states(self,emb,idx_batch=0):
        num_leaves = tf.squeeze(tf.gather(self.num_leaves,idx_batch))
        n_inodes = tf.gather(self.n_inodes,idx_batch)
        embx=tf.gather(tf.gather(emb,idx_batch),tf.range(num_leaves))
        treestr=tf.gather(self.treestr,idx_batch)
        treestr=tf.gather(treestr,tf.range(n_inodes))
        #treestr [n_inodes, 2]
        #[num_leaves, 2*hidden_dim]
        leaf_hc = self.process_leafs(embx)
        leaf_h,leaf_c=tf.split(axis=1,num_or_size_splits=2,value=leaf_hc)
        nodes_h=tf.identity(leaf_h)
        #[num_leaves, hidden_dim]
        nodes_c=tf.identity(leaf_c)
        idx_var=tf.constant(0) #tf.Variable(0,trainable=False)
        with tf.variable_scope("btp_Composition",reuse=True):
            # cW 2*hidden（两个子节点的Hidden value, 5*hidden
            cW = tf.get_variable("cW",[self.degree*self.hidden_dim,(self.degree+3)*self.hidden_dim])
            cb = tf.get_variable("cb",[4*self.hidden_dim])            
            bu,bo,bi,bf=tf.split(axis=0,num_or_size_splits=4,value=cb)
            def _recurrence(node_h,node_c,idx_var):
                node_info=tf.gather(treestr,idx_var)
                #node_info [2, ]
                child_h=tf.gather(node_h,node_info)
                child_c=tf.gather(node_c,node_info)
                flat_ = tf.reshape(child_h,[-1])
                #[1* hidden_dim]
                tmp=tf.matmul(tf.expand_dims(flat_,0),cW)                
                u,o,i,fl,fr=tf.split(axis=1,num_or_size_splits=5,value=tmp)                
                i=tf.nn.sigmoid(i+bi)
                o=tf.nn.sigmoid(o+bo)
                u=tf.nn.tanh(u+bu)
                fl=tf.nn.sigmoid(fl+bf)
                fr=tf.nn.sigmoid(fr+bf)

                f=tf.concat(axis=0,values=[fl,fr])
                c = i * u + tf.reduce_sum(f*child_c,[0])
                h = o * tf.nn.tanh(c)
                node_h = tf.concat(axis=0,values=[node_h,h])
                node_c = tf.concat(axis=0,values=[node_c,c])
                idx_var=tf.add(idx_var,1)
                return node_h,node_c,idx_var
            #Returns the truth value of (x < y) element-wise
            loop_cond = lambda a1,b1,idx_var: tf.less(idx_var,n_inodes)
            loop_vars=[nodes_h,nodes_c,idx_var]
            nodes_h,nodes_c,idx_var=tf.while_loop(loop_cond, _recurrence,
                                                loop_vars,parallel_iterations=10)
            return tf.expand_dims(nodes_h,0)
        #[1* node_num ,hidden_value]
    def add_training_op(self):
        pass

class context_top_down_lstm(object):
    def __init__(self,config, c_bp_lstm, roots_states):
        self.max_sentence_num=20
        self.emb_dim=config.emb_dim
        self.hidden_dim=config.hidden_dim
        self.num_emb=config.num_emb
        self.config=config
        self.sentences_root_hs=roots_states  #[sentence_num , 2*hidden_dim]
        self.reg=config.reg
        self.degree=config.degree
        self.sentence_num=c_bp_lstm.sentence_num
        self.add_placeholders()
        emb_leaves = self.add_embedding()
        self.add_more_variables()
        self.sentences_hidden_states=self.compute_sentences_states_h(emb_leaves)
        #self.sentences_root_states=self.get_root_states(self.sentences_states)
    def get_root_states(self, sentences_states):
        def _get_root_states(x):
            states=tf.gather(x, tf.subtract(tf.gather(tf.shape(x),0),1))
            return states
        hidden_states = tf.map_fn(_get_root_states,sentences_states)        
    def add_embedding(self):
        with tf.variable_scope("Embed",reuse=True):
            #emb_tree [sentence_num, maxnodesize, emb_dim] 
            #input[sentence_num, maxnodesize ]
            embedding=tf.get_variable('embedding')
            tix=tf.to_int32(tf.not_equal(self.t_input,-1))*self.t_input
            emb_tree=tf.nn.embedding_lookup(embedding, tix)
            #sentencenum*maxnodesize*embedding_dim
            emb_tree=emb_tree*(tf.expand_dims(
                        tf.to_float(tf.not_equal(self.t_input,-1)),2))
            return emb_tree
    def add_placeholders(self):
        dim2=self.config.maxnodesize
        self.t_input=tf.placeholder(tf.int32,[None,dim2],name='context_td_input')
        self.t_input=tf.gather(self.t_input, tf.range(self.sentence_num))
        self.t_treestr = tf.placeholder(tf.int32,[None,dim2],name='context_td_tree')
        self.t_treestr=tf.gather(self.t_treestr, tf.range(self.sentence_num))
        self.t_par_leaf = tf.placeholder(tf.int32,[None,dim2],name='context_td_par_leaf')
        self.t_par_leaf = tf.gather(self.t_par_leaf, tf.range(self.sentence_num))

        self.dropout = tf.placeholder(tf.float32,name='context_td_dropout')
        self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.t_input,-1)),[1])
        self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.t_treestr,-1)),[1])
        adder=tf.ones(tf.shape(self.num_leaves),dtype=tf.int32)
        self.n_inodes =tf.add(self.n_inodes,adder)
    def add_more_variables(self):
        with tf.variable_scope('context_td_composition',initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):
            #hidden states and cell states of parents
            cW = tf.get_variable("cW",[self.hidden_dim+self.emb_dim,4*self.hidden_dim],
                initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim)))
            cb = tf.get_variable("cb",[4*self.hidden_dim],
                initializer=tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))
    def calc_wt_init(self,fan_in=300):
        eps=1.0/np.sqrt(fan_in)
        return eps
    def compute_sentences_states_h(self,emb_leaves):
        #return [sentence_num ,nodes_size, hidden_dim]
        inodes_h, inodes_c=self.compute_inodes_states(0)
        nodes_h,nodes_c=self.process_leafs(inodes_h, inodes_c, emb_leaves,0)
        nodes_h_states=tf.expand_dims(nodes_h,axis=0)
        logging.warn('{}'.format(nodes_h_states.shape))
        logging.warn('expand_dims done')
        #[1 nodenum hiddenvalue]
        idx_curbatch=tf.constant(1)
        def _tdcomputestate(idx_curbatch,nodes_h_states):
            tmpinodes_h, tmpinodes_c=self.compute_inodes_states(idx_curbatch)
            tmpnodes_h,tmpnodes_c=self.process_leafs(tmpinodes_h, tmpinodes_c, emb_leaves, idx_curbatch)
            curnodes_h=tf.expand_dims(tmpnodes_h,0)
            nodes_h_states=tf.concat([nodes_h_states, curnodes_h], axis=0)
            idx_curbatch=tf.add(idx_curbatch,1)
            return idx_curbatch, nodes_h_states
        loop_cond=lambda idx,a: tf.less(idx, self.sentence_num)
        loop_vars=[idx_curbatch, nodes_h_states]
        idx_curbatch, nodes_h_states=tf.while_loop(loop_cond, _tdcomputestate, loop_vars,
            shape_invariants=[idx_curbatch.get_shape(),tf.TensorShape([None,None,self.hidden_dim])])
        return nodes_h_states

    def process_leafs(self,inodes_h,inodes_c,emb_leaves,idx_batch):
        logging.warn('begin get num leaves')
        num_leaves = tf.squeeze(tf.gather(self.num_leaves,idx_batch))
        logging.warn('get num leaves done')
        embx=tf.gather(tf.gather(emb_leaves,idx_batch),tf.range(num_leaves))
        logging.warn('get leaf embedding done')
        leaf_parent=tf.gather(tf.gather(self.t_par_leaf,idx_batch),tf.range(num_leaves))
        logging.warn('get leaf parents array done')
        node_h=tf.identity(inodes_h)
        node_c=tf.identity(inodes_c)
        with tf.variable_scope('td_Composition',reuse=True):
            cW=tf.get_variable('cW',[self.hidden_dim+self.emb_dim,4*self.hidden_dim])
            cb=tf.get_variable('cb',[4*self.hidden_dim])
            bu,bo,bi,bf=tf.split(axis=0,num_or_size_splits=4,value=cb)
            idx_var=tf.constant(0)
            logging.warn('begin enumerate the idx_var')
            def _recurceleaf(node_h, node_c,idx_var):
                node_info=tf.gather(leaf_parent, idx_var)
                logging.warn('get t_idx, the index of parent')
                cur_embed=tf.gather(embx, idx_var)
                #node_h:[inode_size, dim_hidden]
                parent_h=tf.gather(node_h, node_info)
                parent_c=tf.gather(node_c, node_info)
                cur_input=tf.concat(values=[parent_h, cur_embed],axis=0)
                flat_=tf.reshape(cur_input, [-1])

                tmp=tf.matmul(tf.expand_dims(flat_,0),cW)

                u,o,i,f=tf.split(axis=1,num_or_size_splits=4,value=tmp)
                i=tf.nn.sigmoid(i+bi)
                o=tf.nn.sigmoid(o+bo)
                u=tf.nn.sigmoid(u+bu)
                f=tf.nn.sigmoid(f+bf)
                c=i*u+tf.reduce_sum(f*parent_c,[0])
                h=o*tf.nn.tanh(c)

                node_h=tf.concat(axis=0,values=[node_h,h])
                node_c=tf.concat(axis=0,values=[node_c,c])
                idx_var=tf.add(idx_var,1)
                logging.warn('get new node_h and new node_c done')
                return node_h, node_c, idx_var
            loop_cond=lambda a1,b1,idx_var:tf.less(idx_var,num_leaves)
            loop_vars=[node_h,node_c,idx_var]
            node_h,node_c,idx_var=tf.while_loop(loop_cond, _recurceleaf,loop_vars,
                shape_invariants=[tf.TensorShape([None,self.hidden_dim]),tf.TensorShape([None,self.hidden_dim]),idx_var.get_shape()])
            logging.warn('return new node_h, finished')
            return node_h,node_c
    def compute_inodes_states(self,idx_batch=0):
        #return [nodes_size, hidden_dim], [nodes_size, cell_dim]
        n_inodes = tf.gather(self.n_inodes,idx_batch)
        t_treestr=tf.gather(tf.gather(self.t_treestr,idx_batch),tf.range(n_inodes))
        #t_treestr[n_inodes]
        node_states = tf.gather(self.sentences_root_hs,idx_batch)
        #[2* hidden_dim]
        root_state, root_cell =tf.split(node_states, num_or_size_splits=2, axis=0)
        root_state=tf.expand_dims(root_state, 0)
        root_cell=tf.expand_dims(root_cell, 0)
        inode_h=tf.identity(root_state)
        inode_c=tf.identity(root_state)
        idx_var=tf.constant(1)
        with tf.variable_scope('context_td_composition',reuse=True):
            cW=tf.get_variable('cW',[self.hidden_dim+self.emb_dim,4*self.hidden_dim])
            cW,_=tf.split(value=cW,num_or_size_splits=[self.hidden_dim, self.emb_dim],axis=0)
            cb=tf.get_variable('cb',[4*self.hidden_dim])
            bu, bo, bi, bf=tf.split(axis=0,num_or_size_splits=4,value=cb)
            def _recurrence(node_h,node_c,idx_var):
                node_info=tf.gather(t_treestr, idx_var) #get t_idx, the index of parent
                parent_h=tf.gather(node_h, node_info)
                parent_c=tf.gather(node_c, node_info)

                flat_=tf.reshape(parent_h, [-1])
                tmp=tf.matmul(tf.expand_dims(flat_,0),cW)
                u,o,i,f=tf.split(axis=1,num_or_size_splits=4,value=tmp)
                i=tf.nn.sigmoid(i+bi)
                o=tf.nn.sigmoid(o+bo)
                u=tf.nn.sigmoid(u+bu)
                f=tf.nn.sigmoid(f+bf)
                c=i*u+tf.reduce_sum(f*parent_c,[0])
                h=o*tf.nn.tanh(c)
                node_h=tf.concat(axis=0,values=[node_h,h])
                node_c=tf.concat(axis=0,values=[node_c,c])
                idx_var=tf.add(idx_var,1)
                return node_h, node_c, idx_var
            loop_cond=lambda a1,b1,idx_var: tf.less(idx_var, n_inodes)
            loop_vars=[inode_h,inode_c,idx_var]
            inode_h,inode_c,idx_var=tf.while_loop(loop_cond, _recurrence,loop_vars,
                shape_invariants=[tf.TensorShape([None, self.hidden_dim]),tf.TensorShape([None,self.hidden_dim]), idx_var.get_shape()])
            return inode_h,inode_c

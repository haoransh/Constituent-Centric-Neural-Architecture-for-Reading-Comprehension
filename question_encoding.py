import numpy as np
import tensorflow as tf
import os
import logging
import load_data
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib import legacy_seq2seq

class question_encoding(object):
    def __init__(self,config):
        logging.warn('begin initialize the question_encoding')
        self.bp_lstm=bottom_up_lstm(config)
        self.td_lstm=top_down_lstm(config,self.bp_lstm)
        #[batch_size,cur_node_num,hidden_value]
        self.bp_states_h=self.bp_lstm.states_h
        logging.warn('the question bp_states:'.format(self.bp_states_h.get_shape()))
        self.td_states_h=self.td_lstm.states_h
        self.config=config
        self.nodes_states=self.get_tree_states(self.bp_states_h, self.td_states_h)
        logging.warn('the question_encoding initialization finished')
    def train(self,data,sess):
        #data has no batch
        logging.warn('data length:{}'.format(len(data)))
        for curidx in range(len(data)):
            question_data=data[curidx][0]
            context_data=data[curidx][2]
            #batch_data is the root of the tree
            b_input, b_treestr, t_input, t_treestr, t_parent=load_data.extract_filled_tree(question_data,self.config.maxnodesize,word2idx=self.config.word2idx)
            context_inputs=[]
            for i in range(len(context_data)):
                c_input,c_treestr,c_input,c_treestr, c_parent=load_data.extract_filled_tree(context_data[i], self.config,maxnodesize, word2idx=self.word2idx)
                context_inputs.append([c_input,c_treestr,c_input, c_treestr,c_parent])
                
            feed={self.bp_lstm.input:b_input, self.bp_lstm.treestr:b_treestr, 
                self.td_lstm.t_input:t_input, self.td_lstm.t_treestr:t_treestr, self.td_lstm.t_par_leaf:t_parent, 
                self.bp_lstm.dropout:self.config.dropout, self.td_lstm.dropout:self.config.dropout}
            nodes_states=sess.run(self.nodes_states,feed_dict=feed)
            logging.warn('nodes_states_shape:{}'.format(nodes_states.shape))
        return nodes_states
        #states: [batch_num, nodes_num, hidden_dim]
    def get_tree_states(self, bp_states, td_states):
        #bp_states[nodesize * hidden_dim]
        rev_td_states=tf.reverse(td_states,axis=[0])
        states=tf.concat(values=[bp_states, rev_td_states],axis=1)
        return states

        

class top_down_lstm(object):
    def __init__(self,config,bp_lstm):
        self.emb_dim=config.emb_dim
        self.hidden_dim=config.hidden_dim
        self.num_emb=config.num_emb
        self.config=config        
        self.nodes_hs=bp_lstm.states_h
        self.nodes_cs=bp_lstm.states_c
        self.reg=config.reg
        self.degree=config.degree
        self.add_placeholders()
        emb_leaves = self.add_embedding()
        self.add_more_variables()
        self.states_h=self.compute_states(emb_leaves)
        self.states_h=tf.reshape(self.states_h,[self.n_inodes+self.num_leaves, self.hidden_dim])
        #return all nodes information
    def add_embedding(self):
        with tf.variable_scope("Embed",reuse=True):
            #emb_tree [maxnodesize, emb_dim] 
            #multiplier: [maxnodesize * 1 ]
            embedding=tf.get_variable('embedding')
            tix=tf.to_int32(tf.not_equal(self.t_input,-1))*self.t_input
            emb_tree=tf.nn.embedding_lookup(embedding, tix)
            emb_tree=emb_tree*(tf.expand_dims(
                        tf.to_float(tf.not_equal(self.t_input,-1)),1))
            return emb_tree
    def add_placeholders(self):
        dim2=self.config.maxnodesize
        self.t_input=tf.placeholder(tf.int32,[dim2],name='td_input')
        self.t_treestr = tf.placeholder(tf.int32,[dim2],name='td_tree')
        self.t_par_leaf = tf.placeholder(tf.int32,[dim2],name='td_par_leaf')
        self.dropout = tf.placeholder(tf.float32,name='td_dropout')
        self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.t_input,-1)),[0])
        self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.t_treestr,-1)),[0])
        self.n_inodes =tf.add(self.n_inodes,1) #consider the parent of root is -1
    def add_more_variables(self):
        with tf.variable_scope('td_Composition',initializer=tf.contrib.layers.xavier_initializer(),  \
            regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):
            #hidden states and cell states of parents
            cW = tf.get_variable("cW",[self.hidden_dim+self.emb_dim,4*self.hidden_dim],initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim)))
            cb = tf.get_variable("cb",[4*self.hidden_dim],initializer=tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))
    def calc_wt_init(self,fan_in=300):
        eps=1.0/np.sqrt(fan_in)
        return eps
    def compute_states(self,emb_leaves):
        inodes_h, inodes_c=self.compute_inodes_states()
        nodes_h,nodes_c=self.process_leafs(inodes_h, inodes_c, emb_leaves)
        logging.warn('process leaves done')
  
        return nodes_h

    def process_leafs(self,inodes_h,inodes_c,emb_leaves):
        num_leaves = self.num_leaves
        embx=tf.gather(emb_leaves,tf.range(num_leaves))
        leaf_parent=tf.gather(self.t_par_leaf,tf.range(num_leaves))
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
                cur_embed=tf.gather(embx, idx_var)
                #initial node_h:[inode_size, dim_hidden]
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
                return node_h, node_c, idx_var
            loop_cond=lambda a1,b1,idx_var:tf.less(idx_var,num_leaves)
            loop_vars=[node_h,node_c,idx_var]
            node_h,node_c,idx_var=tf.while_loop(loop_cond, _recurceleaf,loop_vars,shape_invariants=[tf.TensorShape([None,self.hidden_dim]),tf.TensorShape([None,self.hidden_dim]),idx_var.get_shape()])
            logging.warn('return new node_h, finished')
            return node_h,node_c
    def compute_inodes_states(self):
        n_inodes = self.n_inodes
        t_treestr=tf.gather(self.t_treestr,tf.range(n_inodes))
        #node_states [inode_size, dim_hidden]
        root_state=tf.gather(self.nodes_hs,tf.subtract(tf.gather(tf.shape(self.nodes_hs),0),1))
        root_cell=tf.gather(self.nodes_cs,tf.subtract(tf.gather(tf.shape(self.nodes_cs),0),1))
        
        root_state=tf.expand_dims(root_state,0)
        root_cell=tf.expand_dims(root_cell,0)
        inode_h=tf.identity(root_state)
        inode_c=tf.identity(root_cell)
        idx_var=tf.constant(1)  
        with tf.variable_scope('td_Composition',reuse=True):
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
            inode_h,inode_c,idx_var=tf.while_loop(loop_cond, _recurrence,loop_vars,shape_invariants=[tf.TensorShape([None, self.hidden_dim]),tf.TensorShape([None,self.hidden_dim]), idx_var.get_shape()])
            return inode_h,inode_c

class bottom_up_lstm(object):
    def __init__(self,config):
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.num_emb = config.num_emb
        self.config=config
        self.reg=self.config.reg  #regulizer parameter
        self.degree=config.degree  #  2, the N-ary
        self.add_placeholders()
        #maxnodesize * emb_dim
        emb_leaves = self.add_embedding()
        self.add_model_variables()
        self.states_h, self.states_c = self.compute_states(emb_leaves)
        self.states_h=tf.reshape(self.states_h,[self.n_inodes+self.num_leaves, self.hidden_dim])
        self.states_c=tf.reshape(self.states_c,[self.n_inodes+self.num_leaves, self.hidden_dim])
        #[node_num ,hidden_value]
        #batch_states A tensor list: [batch_size, cur_node_num, hidden_value] node_num: include leaves and internal nodes

        #or we can choose to load embedding from glove
        #self.emb_mat=np.array([idx2vec[idx] if idx in idx2vec  \
        #    else np.random.multivariate_normal(np.zeros(config.emb_dim) for idx in range(len(config.vocab_counter))       
    def add_placeholders(self):       
        dim2=self.config.maxnodesize #parse tree node鐨勬暟閲�
        self.input = tf.placeholder(tf.int32,[dim2],name='input')
        self.treestr = tf.placeholder(tf.int32,[dim2,2],name='tree')     
        self.dropout = tf.placeholder(tf.float32,name='dropout')
        self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.treestr,-1)),[0,1])
        #瀵逛竴涓狟atch涔嬪唴鐨勮繘琛屾灇涓緎um
        self.n_inodes = self.n_inodes//2
        self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input,-1)),[0])
    def add_embedding(self):
        #璁剧疆涓篻love鐨別mbedding
        #embed=np.load('glove{0}_uniform.npy'.format(self.emb_dim))
        with tf.variable_scope("Embed",regularizer=None):
            #embedding=tf.get_variable('embedding',[self.num_emb,self.emb_dim],initializer=self.emb_mat, trainable=False)
            embedding=tf.get_variable('embedding',initializer=self.config.embedding,trainable=False,regularizer=None)
            ix=tf.to_int32(tf.not_equal(self.input,-1))*self.input
            emb_tree=tf.nn.embedding_lookup(embedding,ix)
            #emb_tree [maxnodesize, emb_dim] 
            #multiplier: [maxnodesize * 1 ]
            emb_tree=emb_tree*(tf.expand_dims(
                        tf.to_float(tf.not_equal(self.input,-1)),1))
            return emb_tree
    def calc_wt_init(self,fan_in=300):
        eps=1.0/np.sqrt(fan_in)
        return eps
    def add_model_variables(self):

        with tf.variable_scope("btp_Composition",
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
            #鍙跺瓙鑺傜偣娌℃湁input gate鍜宖orget gate,闇�瑕佽绠梠utput gate 鍜孖nput value
            #鍙朿b鐨勫墠 2*hidde_dim 缁�           
            #x [emb_dim]
            def _recurseleaf(x):
                #[1, emb_dim], [emb_dim, 2*self.hidden_dim]
                concat_uo = tf.matmul(tf.expand_dims(x,0),cU) + b
                #鎶奵oncat_uo鍒囧壊鎴�
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

    def compute_states(self,emb):
        #闄嶇淮搴�
        num_leaves = self.num_leaves
        n_inodes = self.n_inodes
        embx=tf.gather(emb,tf.range(num_leaves))
        treestr=tf.gather(self.treestr,tf.range(n_inodes))
        #treestr [n_inodes, 1 or 2]
        #[num_leaves, 2*hidden_dim]
        leaf_hc = self.process_leafs(embx)
        leaf_h,leaf_c=tf.split(axis=1,num_or_size_splits=2,value=leaf_hc)
        nodes_h=tf.identity(leaf_h)
        #[num_leaves, hidden_dim]
        nodes_c=tf.identity(leaf_c)
        idx_var=tf.constant(0) #tf.Variable(0,trainable=False)
        with tf.variable_scope("btp_Composition",reuse=True):
            # cW 2*hidden锛堜袱涓瓙鑺傜偣鐨凥idden value, 5*hidden
            cW = tf.get_variable("cW",[self.degree*self.hidden_dim,(self.degree+3)*self.hidden_dim])
            cb = tf.get_variable("cb",[4*self.hidden_dim])            
            bu,bo,bi,bf=tf.split(axis=0,num_or_size_splits=4,value=cb)
            def _recurrence(node_h,node_c,idx_var):
                #鏍规嵁index浠嶯ode_h鍜宯ode_c涓彇鍊硷紝骞朵笉鏂洿鏂癗ode_h鍜孨ode_c锛屼繚璇佸彇鐨勯『搴�
                node_info=tf.gather(treestr,idx_var)
                #node_info shape [2, ]
                child_h=tf.gather(node_h,node_info)
                child_c=tf.gather(node_c,node_info)
                flat_ = tf.reshape(child_h,[-1])
                #灞曟垚1-D vector 
                #[1* hidden_dim
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
            return nodes_h,nodes_c
        #[node_num ,hidden_value]
    def add_training_op(self):
        pass

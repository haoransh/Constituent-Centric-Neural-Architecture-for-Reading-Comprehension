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
from context_encoding import *
class ccrc_model(object):
    #Here I assume that 
    #the output size of LSTM unit used in the answer generation step is config.hidden_dim

    def __init__(self, config):
        self.q_encoding=question_encoding(config)
        self.c_encoding=context_encoding(config)
        self.config=config
        ##to do list
        self.att_layer=attentioned_layer(self.q_encoding, self.c_encoding)
        #self.max_candidate_answers=config.max_candidate_answers
        #self.answer_generation=answer_genaration() #need Tensors from q_encoding, c_endoing, att_layer
        self.projection_input_dim=config.hidden_dim
        #predictions=self.answer_generation.predicted_list
        #answer=self.answer_generation.correct_answer
        self.add_variables()
        #self.loss=self.get_loss(predictions,answer)
        #self.train_op=self.add_training_op()
    def train(self,data,sess):
        #data has no batch
        #the candidate answer constituency should be processed before feed
        logging.warn('data length:{}'.format(len(data)))
        for curidx in range(len(data)):
            question_data=data[curidx][0]
            answer_data=data[curidx][1][0] #consider that in train dataset, there is only one ground-truth answer for each question.
            context_data=data[curidx][2]
            #context_data is the list of root of sentences
            b_input, b_treestr, t_input, t_treestr, t_parent=load_data.extract_filled_tree(question_data,self.config.maxnodesize,word2idx=self.config.word2idx)
            c_inputs,c_treestrs. c_t_inputs,c_t_treestrs,c_t_parents=[],[],[],[],[]
            for i in range(len(context_data)):
                c_input,c_treestr,c_t_input,c_t_treestr, c_t_parent=load_data.extract_filled_tree(context_data[i], self.config,maxnodesize, word2idx=self.word2idx)
                c_inputs.append(c_input)
                c_treestrs.append(c_treestr)
                c_t_inputs.append(c_t_input)
                c_t_treestrs.append(c_t_treestr)
                c_t_parents.append(c_t_parent)

            feed={
                self.q_encoding.bp_lstm.input:b_input,
                self.q_encoding.bp_lstm.treestr:b_treestr, 
                self.q_encoding.td_lstm.t_input:t_input, 
                self.q_encoding.td_lstm.t_treestr:t_treestr, 
                self.q_encoding.td_lstm.t_par_leaf:t_parent, 
                self.q_encoding.bp_lstm.dropout:self.config.dropout, 
                self.q_encoding.td_lstm.dropout:self.config.dropout,

                self.c_encoding.c_bp_lstm.sentence_num:sentence_num, 
                self.c_encoding.c_bp_lstm.input:c_inputs, 
                self.c.encoding.c_bp_lstm.treestr:c_treestrs, 
                self.c_encoding.c_bp_lstm.dropout:self.config.dropout,
                self.c_encoding.c_td_lstm.t_input:c_t_inputs,
                self.c_encoding.c_td_lstm.t_treestr:c_t_treestrs,
                self.c_encoding.c_td_lstm.t_par_leaf:c_t_parents,
                self.c_encoding.c_td_lstm.dropout:self.config.dropout
                }
            nodes_states=sess.run(self.nodes_states,feed_dict=feed)
            logging.warn('curidx:{}'.format(curidx))
            logging.warn('nodes_states:{}'.format(nodes_states))
            logging.warn('nodes_states_shape:{}'.format(nodes_states.shape))
        return nodes_states
    def add_variables(self):
        with tf.variable_scope('projection_layer'):
            softmax_W=tf.get_variable('softmax_w',[self.config.hidden_dim, 1],initializer=tf.random_normal_initializer(mean=0, stddev=1/self.config.hidden_dim))
            softmax_b=tf.get_variable('softmax_b',[1], initializer=tf.constant_initializer(0.0))
        self.global_step=tf.Variable(0, name='global_step', trainable=False)
    def add_training_op(self):
        opt=tr.train.AdagradOptimizer(self.config.lr)
        train_op=opt.minimize(self.loss)
        return train_op
        #self.learning_rate=tf.maximun(1e-5, tf.train.exponential_devay(config.learning_rate, cinfig.global_step, config.lr_deday_steps, config.))

    def get_loss(self, predictions, answer):
        #predictions: [candidate_answer_num,  hidden_dim]
        with tf.variable_scope('projection_layer',reuse=True):
            softmax_w=tf.get_variable('softmax_w')
            softmax_b=tf.get_variable('softmax_b')
            scores=tf.matmul(predictions, softmax_w)+softmax_b
            scores=tf.squeeze(scores) #[candidate_answer_num]
            scores=tf.expand_dims(scores, 0) #[1, candidate_answer_num]
            truth=tf.onehot(answer, tf.gather(tf.shape(predictions),0))
            truth=tf.expand_dims(truth, 0)
            cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=truth)
            cross_entropy=tf.squeeze(cross_entropy)
            return cross_entropy

class answer_generation():
    def __init__(self):
        self.predicted_list=None
class attentioned_layer(object):
    def __init__(self, question_encode, context_encode):
        self.attentioned_hidden_states=self.get_context_attentioned_hiddens
        #[sentence_num, node_size, 4*hidden_dim]

    def get_context_attentioned_hiddens(self, question_encode, context_encode):
        context_constituency=context_encode.sentences_final_states
        #[sentence_num, node_size, 2* hidden_dim]
        question_constituency=question_encode.nodes_states
        question_leaves=question_encode.bp_lstm.num_leaves
        question_treestr=question_encode.bp_lstm.treestr
        #[node_size, hidden_dim]
        sentence_constituency=tf.gather(context_constituency,0)
        context_attentioned_hiddens=self.get_sentence_attention_values(sentence_constituency, question_constituency, question_leaves, question_treestr)
        context_attentioned_hiddens=tf.expand_dims(context_attentioned_hiddens)
        sentence_num=context_encode.sentence_num
        sentence_id=tf.constant(1)
        def _recurse_sentence(final_hiddens, sentence_id):
            sentence_constituency=tf.gather(context_constituency, sentence_id)
            cur_sentence_states=self.get_sentence_attention_values(sentence_constituency, question_constituency, question_leaves, question_treestr)
            cur_sentence_states=tf.expand_dims(cur_sentence_states, axis=0)
            final_hiddens=tf.concat([final_hiddens, cur_sentence_states],axis=0)
            sentence_id=tf.add(sentence_id, 1)
            return final_hiddens, sentence_id
        loop_cond=lambda a1,a2, sentence_idx:tf.less(sentence_id, sentence_num) 
        loop_vars=[context_attentioned_hiddens,sentence_id]
        context_attentioned_hiddens, sentence_id=tf.while_loop(loop_cond, _recurse_sentence, loop_vars,
            shape_invariants=[tf.TensorShape(None,None,4*self.config.hidden_dim),sentence_id.get_shape()])
        return attentioned_hiddens
        #[sentence_num, node_size, 4*hidden_dim]
        #context_constituency_num=tf.shape(context_constituency)
        # loop all the sentences
        # loop all the constituency in one sentence
        # in loop get all the representation of constituency in the context
        # concate it the original representation generated by context_encode class
    def get_sentence_attention_values(self, sentence_constituency, question_constituency, question_leaves, question_treestr):
        # return [nodes_num, 4*hidden_dim]
        context_constituency=tf.gather(sentence_constituency,0)
        attentioned_hiddens=self.get_constituency_attention_values(context_constituency, question_constituency, question_leaves, question_treestr)
        attentioned_hiddens=tf.expand_dims(attentioned_hiddens, 0)
        sentence_nodes_num=tf.gather(tf.shape(sentence_constituency),0)
        idx_var=tf.constant(1)
        def _recurse_context_constituency(attentioned_hiddens, idx_var):
            context_constituency=tf.gather(sentence_constituency, idx_var)
            cur_constituency_attentioned_hiddens=self.get_constituency_attention_values(context_constituency, question_constituency, question_leaves, question_treestr)
            cur_constituency_attentioned_hiddens=tf.expand_dims(attentioned_hiddens, 0)
            attentioned_hiddens=tf.concat([attentioned_hiddens, cur_constituency_attentioned_hiddens],axis=0)
            idx_var=tf.add(idx_var,1)
            return attentioned_hiddens, idx_var

        loop_cond=lambda a1,idx:tf.less(idx, sentence_nodes_sum)
        loop_vars=[attentioned_hiddens, idx_var]
        attentioned_hiddens, idx_var=tf.while_loop(loop_cond,_recurse_context_constituency, loop_vars, 
            shape_invariants=[tf.TensorShape(None, 4*self.hidden_dim), idx_var.get_shape()])

        return attentioned_hiddens

    def get_constituency_attention_values(self, context_constituency, question_constituency, question_leaves, question_treestr):
        #return [4*hidden_dim]
        #context_constituency: [2* hidden_dim]
        q_nodes=tf.gather(tf.shape(question_constituency),0)
        q_allnodes=tf.range(q_nodes)
        def _get_score(inx):
            q_node_hiddens=tf.gather(question_constituency, inx) #[2*hidden_dim]
            attention_score=tf.reduce_sum(tf.multiply(q_node_hiddens,context_constituency))
            return attention_score
        nodes_attentions=tf.map_fn(_get_score, q_nodes)
        #################neet normalize the attention scores
        q_leaves=tf.range(question_leaves)
        def _get_attentional_leaves(inx):
            hiddens=tf.gather(question_constituency, inx) #[2*hidden_dim]
            attention_score=tf.gather(nodes_attentions, inx)
            attentional_leaves=tf.multiply(hiddens, attention_score)
            return attentional_leaves
        attentional_representations=tf.map_fn(_get_attentional_leaves, q_leaves)
        inodes_num=tf.substract(q_nodes,question_leaves)
        idx_var=tf.constant(0)
        def _recurse_q_nodes(attentional_representations, idx_var):
            node_idx=tf.add(idx_var, question_leaves)
            node_attentional_score=tf.gather(nodes_attentions, node_idx)

            node_hidden=tf.gather()
            node_children=tf.gather(question_treestr, idx_var) #[2]
            children_attention_score=tf.gather(nodes_attentions, node_children) #[2]
            children_attention_score=tf.nn.softmax(children_attention_score)
            children_attention_score=tf.expand_dims(children_attention_score, axis=0)
            children_attentional_rep=tf.gather(attentional_representations, node_children) #[2, 2*hidden_dim]
            children_combine=tf.matmul(children_attention_score, children_attentional_rep) #[1, 2*hidden_dim]
            children_combine=tf.squeeze(children_combine) #[2* hidden_dim]
            b=tf.multiply(tf.add(children_combine, context_constituency), node_attentional_score) #[2* hidden_dim]
            b=tf.expand_dims(b,axis=0)
            attentional_representations=tf.concat([attentional_representations, b], axis=0)
            idx_var=tf.add(idx_var, 1)
            return attentional_reprensentations, idx_var
        loop_cond=lambda a1, idx:tf.less(idx, inodes_num)
        loop_vars=[attentional_representations, idx_var]
        attentional_representations, idx_var=tf.while_loop(loop_cond, _recurse_q_nodes,loop_vars, 
            shape_invariants=[tf.TensorShape(None, 2*self.hidden_dim), idx_var.get_shape()])
        root_attentional_representation=tf.gather(attentional_representations, tf.substract(q_nodes,1))
        concated_attentional_rep=tf.cancat([constituency, root_attentional_representation], axis=0)
        return concated_attentional_rep

if __name__=='__main__':
    from my_main import Config
    config=Config()
    word2idx,embedding=load_data.load_embedding()
    config.embedding=embedding
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        logging.warn('begin build the model')
        model = ccrc_model(config)
        logging.warn('model build done')
        sess.run(tf.global_variables_initializer())


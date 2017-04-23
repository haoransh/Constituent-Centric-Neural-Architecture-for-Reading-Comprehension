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
from attention_layer import *

class ccrc_model(object):
    #Here I assume that 
    #the output size of LSTM unit used in the answer generation step is config.hidden_dim

    def __init__(self, config):
        self.q_encoding=question_encoding(config)
        self.c_encoding=context_encoding(config)
        self.config=config
        self.sentence_num=self.c_encoding.sentence_num
        ##to do list
        self.att_layer=attentioned_layer(config, self.q_encoding, self.c_encoding)
        self.scope_index=0
        #every constituency has a representation [ 4* hidden_dim]
        with tf.variable_scope('candidate_answer_generation_forward'):
            self.fwcell=rnn.BasicLSTMCell(self.config.hidden_dim, activation=tf.nn.tanh)
        with tf.variable_scope('candidate_answer_generation_backword'):
            self.bwcell=rnn.BasicLSTMCell(self.config.hidden_dim, activation=tf.nn.tanh)
        self._fw_initial_state=self.fwcell.zero_state(1,dtype=tf.float32)
        self._bw_initial_state=self.bwcell.zero_state(1,dtype=tf.float32)
        self.add_placeholders()
        self.candidate_answer_representations=self.get_candidate_answer_representations()
        assert tf.gather(tf.shape(self.candidate_answer_representations),0)==self.candidate_answer_overall_number
        self.loss=self.get_loss(self.candidate_answer_representations,self.correct_answer_idx)
        self.train_op=self.add_training_op()
    def add_placeholders(self):
        self.correct_answer_idx=tf.placeholder(tf.int32, name='correct_answer_index')
        self.candidate_answers=tf.placeholder(tf.int32, [None,None,None],name='candidate_answers')#[None,None,None]
        self.candidate_answer_overall_number=tf.placeholder(tf.int32,name='candidate_overall_number')

    def get_candidate_answer_representations(self):
        #return answer: a correnct answer index
        #return predictions, 
        candidate_answers=self.candidate_answers #[sentence_num, candidate_number, constituency_idlist]
        sentence_candidate_answers=tf.gather(candidate_answers, 0)
        sentence_attentioned_hidden_states=tf.gather(self.att_layer.attentioned_hidden_states,0)
        candidates_representations=self.get_candidates_representations_in_sentence(sentence_candidate_answers, sentence_attentioned_hidden_states)
        candidates_representations=tf.expand_dims(candidates_representations, 0)
        
        all_sentence_candidates_representations=tf.identity(candidates_representations)
        sentence_num=tf.gather(tf.shape(self.att_layer.attentioned_hidden_states),0)
        logging.warn('attentioned_hidden_states:{}'.format(self.att_layer.attentioned_hidden_states))
        idx_var=tf.constant(1)
        def _recurse_sentence(sentences_candidates_representations, idx_var):
            sentence_candidate_answers=tf.gather(tf.shape(candidate_answers), idx_var)
            sentence_attentioned_hidden_states=tf.gather(self.att_layer.attentioned_hidden_states, idx_var)
            candidates_representations=self.get_candidates_representations_in_sentence(sentence_candidate_answers, sentence_attentioned_hidden_states)
            candidates_representations=tf.expand_dims(candidates_representations, 0)
            sentences_candidates_representations=tf.cancat([sentences_candidates_representations, candidate])
            idx_var=tf.add(idx_var, 1)
            return sentences_candidates_representations, idx_var
        loop_cond=lambda a1, idx: tf.less(idx, sentence_num)
        loop_vars=[all_sentence_candidates_representations, idx_var]
        all_sentence_candidates_representations, idx_var=tf.while_loop(loop_cond, _recurse_sentence,loop_vars, 
            shape_invariants=[tf.TensorShape([None, None, 2*self.config.hidden_dim]), idx_var.get_shape()])

        all_sentence_candidates_representations=tf.reshape(all_sentence_candidates_representations, [-1, 2*self.config.hidden_dim])        
        return all_sentence_candidates_representations
    def get_candidates_representations_in_sentence(self, sentence_candidate_answers, sentence_attentioned_hidden_states):
        candidate_answer_num=tf.gather(tf.shape(sentence_candidate_answers), 0)
        logging.warn('candidate_answer_num:{}'.format(candidate_answer_num))
        logging.warn('sentence_candidate_answers:{}'.format(sentence_candidate_answers))
        candidate_answer_nodeids=tf.gather(sentence_candidate_answers, 0) #a node idx list
        candidate_answer_hidden_list=tf.gather(sentence_attentioned_hidden_states, candidate_answer_nodeids)
        candidate_final_representations=self.get_candidate_answer_final_representations(candidate_answer_hidden_list)
        candidates_final_representations=tf.expand_dims(candidate_final_representations, 0)
        idx_cand=tf.constant(1)
        def _recurse_candidate_answer(candidate_final_representations, idx_cand):
            cur_candidate_answer_nodeids=tf.gather(sentence_candidate_answers, idx_cand)
            cur_candidate_answer_hidden_list=tf.gather(sentence_attentioned_hidden_states, cur_candidate_answer_nodeids)
            cur_candidate_final_representations=tf.expand_dims( 
                self.get_candidate_answer_final_representations(cur_candidate_answer_hidden_list), 0)
            candidate_final_representations=tf.concat([candidate_final_representations, cur_candidate_final_representations], axis=0)
            idx_cand=tf.add(idx_cand,1)
            return candidate_final_representations, idx_cand
        loop_cond=lambda a1,idx:tf.less(idx, candidate_answer_num)
        loop_vars=[candidates_final_representations, idx_cand]
        candidates_final_representations, idx_cand=tf.while_loop(loop_cond, _recurse_candidate_answer, loop_vars,
            shape_invariants=[tf.TensorShape([None, 2*self.config.hidden_dim]),idx_cand.get_shape()])
        return candidates_final_representations
    def get_candidate_answer_final_representations(self, candidate_answer_hidden_list):
        inputs=tf.expand_dims(candidate_answer_hidden_list,axis=0)
        sequence_length=tf.gather(tf.shape(inputs),1)
        sequence_length=tf.expand_dims(sequence_length, 0)
        #with tf.variable_scope('candidate_answer_generation_forward',reuse=True):
        #    fwcell=rnn.BasicLSTMCell(self.config.hidden_dim, activation=tf.nn.tanh) 
        #with tf.variable_scope('candidate_answer_generation_backward',reuse=True):
        #    bwcell=rnn.BasicLSTMCell(self.config.hidden_dim, activation=tf.nn.tanh)
        chain_outputs, chain_state=tf.nn.bidirectional_dynamic_rnn(self.fwcell, self.bwcell, inputs, 
            sequence_length, initial_state_fw=self._fw_initial_state, initial_state_bw=self._bw_initial_state,scope='candidate_answer_{}'.format(self.scope_index))

        self.scope_index+=1
        chain_outputs=tf.concat(chain_outputs, 2)
        chain_outputs=tf.gather(chain_outputs, 0)
        output=tf.gather(chain_outputs, tf.subtract(tf.gather(tf.shape(chain_outputs),0),1))
        return output #[2*hidden_dim]
    def train(self,data,sess):
        #data has no batch
        #the candidate answer constituency should be processed before feed
        logging.warn('data length:{}'.format(len(data)))
        losses=[]
        for curidx in range(len(data)):
            question_data=data[curidx][0]
            answer_data=data[curidx][1][0] #a word idx list
            context_data=data[curidx][2]
            candidate_answers, target_answer_idx, candidate_answers_number=load_data.candidate_answer_generate(answer_data, context_data)
            if not has_answer:
                logging.warn('It has no answer in constituency')
                continue
            #context_data is the list of root of sentences
            b_input, b_treestr, t_input, t_treestr, t_parent=load_data.extract_filled_tree(question_data,self.config.maxnodesize,word2idx=self.config.word2idx)
            c_inputs,c_treestrs, c_t_inputs,c_t_treestrs,c_t_parents=[],[],[],[],[]
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
                self.c_encoding.c_td_lstm.dropout:self.config.dropout,

                self.correct_answer_idx:target_answer_idx,
                self.candidate_answers:candidate_answers,
                self.candidate_answer_overall_number:candidate_ansewrs_number
                }
            fetches=[self.loss, self.train_op]
            curloss, curtrain=sess.run(fetches,feed_dict=feed)
            losses.append(curloss)
            logging.warn('curidx:{}'.format(curidx))
            logging.warn('curl loss:{}'.format(curloss))
        average_loss=np.array(losses).mean()
        return average_loss
    def add_variables(self):
        with tf.variable_scope('projection_layer'):
            softmax_W=tf.get_variable('softmax_w',[2* self.config.hidden_dim, 1],initializer=tf.random_normal_initializer(mean=0, stddev=1/self.config.hidden_dim))
            softmax_b=tf.get_variable('softmax_b',[1], initializer=tf.constant_initializer(0.0))
        self.global_step=tf.Variable(0, name='global_step', trainable=False)

        #self.learning_rate=tf.maximun(1e-5, tf.train.exponential_devay(config.learning_rate, cinfig.global_step, config.lr_deday_steps, config.))
    def get_loss(self, predictions, answer):
        #predictions: [candidate_answer_num,  2* hidden_dim]
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
    def add_training_op(self):
        opt=tr.train.AdagradOptimizer(self.config.lr)
        train_op=opt.minimize(self.loss)
        return train_op
if __name__=='__main__':
    logging.basicConfig(filename="ccrc_model_test.log",level=logging.WARNING)
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


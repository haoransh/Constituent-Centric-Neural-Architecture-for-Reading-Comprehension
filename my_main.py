import os
import numpy as np
import tensorflow as tf
import random
import pickle
import ccrc_model
import pdb
import time
import load_data
import sys
import logging
class Config(object):
    num_emb=None
    emb_dim = 300#300d glove embedding
    hidden_dim = 150
    degree = 2
    num_epochs = 1
    early_stopping = 2
    dropout = 0.5
    lr = 0.05
    num_emb=1000 #default
    emb_lr = 0.1
    reg=0.0001
    batch_size = 5
    maxseqlen = 100
    maxnodesize = 300
    trainable_embeddings=True
    word2idx=None

def train(restore=False):
    config=Config()
    logging.basicConfig(filename="logger.log",level=logging.WARNING)
    data,word2idx,embedding=load_data.load_squad_data()
    #data['train']     [#][0] the root node of the question
    #embedding is for later usage
    config.embedding=embedding
    config.word2idx=word2idx
    assert len(word2idx)==len(embedding)
    num_emb=len(word2idx)
    config.num_emb=num_emb
    config.maxnodesize=200
    config.maxseqlen=100
    random.seed(42)
    np.random.seed(42)
    train=data['train']
    logging.warn('the length of train data:{}'.format(len(train)))
    model=ccrc_model.bi_tree_lstm(config)
    init=tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        start_time=time.time()
        if restore:saver.restore(sess,'./ckpt/tree_rnn_weights')
        bp_states,td_states=model.train(train,sess)
        logging.warn('final_bp_states:{}'.format(bp_states))
        logging.warn('final_bp_states_shape:{}'.format(bp_states.shape))
        logging.warn('final_td_states:{}'.format(td_states))
        logging.warn('final_td_states_shape:{}'.format(td_states.shape))
                    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        restore=True
    else:restore=False
    train(restore)


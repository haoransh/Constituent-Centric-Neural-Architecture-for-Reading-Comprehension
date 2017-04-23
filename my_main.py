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
    max_candidate_answers=100
    word2idx=None

def train(restore=False):
    gpu_config=tf.ConfigProto()
    gpu_config.gpu_options.allow_growth=True
    config=Config()
    logging.basicConfig(filename="logger.log",level=logging.WARNING)
    data,word2idx,embedding=load_data.load_squad_data()
    #data['train']  
    #[#][0] the root node of the question
    #[#][1] one answer: a word index list
    #[#][2] the root node list of the context
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
    model=ccrc_model.ccrc_model(config)
    init=tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        start_time=time.time()
        if restore:saver.restore(sess,'./ckpt/tree_rnn_weights')
        loss=model.train(train,sess)
        print('average loss:{}'.format(loss))
if __name__ == '__main__':
    if len(sys.argv) > 1:
        restore=True
    else:restore=False
    train(restore)


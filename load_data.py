#Attention!!!!! prepro_each only return the first paragraph of the first article for debugging, you can modify the `prepro_each` function according to corresponding comment.

from tqdm import tqdm
import json
import os
import nltk
import argparse
from collections import Counter
import re
import glob
from tf_treenode import tNode,processTree
import random
import numpy as np
import logging
def load_embedding():
    word2idx={}
    embeddings=[]
    args=get_args()
    with open(args.glove_path) as infile:
        for line in tqdm(infile.readlines()):
            array=line.lstrip().rstrip().split(' ')
            vector=list(map(float,array[1:]))
            embeddings.append(vector)
            word=array[0]
            word2idx[word]=len(embeddings)-1 
            #word=array[0]
    return word2idx, embeddings
def load_squad_data():
    args=get_args()
    train_data, trainCounter,dev_data, devCounter=prepro(args)
    train_qlist=[]
    dev_qlist=[]
    with open('train_q.txt','w+')as qout, open('train_a.txt','w+') as aout, open('train_c.txt','w+') as cout:    
        for qindex, data in enumerate(tqdm(train_data)):
            question, answers, context=data[0],data[1],data[2]
            train_qlist.append(question)
            qout.write(str(qindex)+'：'+question+'\n')
            aout.write('\n'.join([str(qindex)+'：'+answer+'\n' for answer in answers]))
            cout.write(str(qindex)+'：'+context+'\n')
    sum_counter=trainCounter+devCounter
    word2idx,embedding=load_embedding()
    with open('vocab.txt','w+') as outfile:
        for i in sum_counter:
            outfile.write(i+'\n')
    train_trees=[]
    train_answer=[]
    train_context_trees=[]
    for i in range(len(train_data)):
        train_trees.append(get_tree(train_data[i][0]))

        train_answer=get_word_idx(word_tokenize(train_data[i][1][0])) #consider that only one correct answer in train dataset

        cur_context_trees=[]
        contexts=nltk.sent_tokenize(train_data[i][2])
        for j in range(len(contexts)):
            cur_context_trees.append(get_tree(contexts[j]))
        train_context_trees.append(cur_context_trees)
        
    for i in range(len(train_data)):
        train_data[i][0]=train_trees[i]
        train_data[i][1]=train_answer
        train_data[i][2]=train_context_trees[i]
    #train_data[#][0] is the root node of one tree
    #train_data[#][1] is the wordidx list of target answer
    #train_data[#][2] is the root list of the sentence
    data={'train':train_data,'dev':dev_data}
    return data, word2idx, embedding

def get_word_idx(word_list, word2idx):
    idx_list=[]
    for word in word_list:
        if word2idx.get(word):
            idx_list.append(word2idx[word])
        elif word2idx.get(word.lower()):
            idx_list.append(word2idx[word.lower()])
        else:
            logging.warn('no wordidx for answer:{}'.format(word))
            idx_list.append(word2idx['unknown'])
    return idx_list

def get_args():
    parser=argparse.ArgumentParser()
    source_dir='/home2/shr/data/nlp/squad'
    target_dir='data/squad'
    glove_path='/home/shr/data/glove/glove.6B.300d.txt'
    parser.add_argument('-s','--source_dir',default=source_dir)
    parser.add_argument('-t','--target_dir',default=target_dir)
    parser.add_argument('-gs','--glove_path',default=glove_path)
    #glove corpus, glove_dir, glove_vec_size
    return parser.parse_args()

def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    train_data, trainCounter = prepro_each(args,'train')
    dev_data, devCounter = prepro_each(args,'dev')
    return train_data, trainCounter, dev_data, devCounter

def prepro_each(args, data_type):
    source_path=os.path.join(args.source_dir,'{}-v1.1.json'.format(data_type))
    source_data=json.load(open(source_path,'r'))
    qlist=[]
    clist=[]
    word_counter=Counter()
    #lower_word_counter=Counter()
    retdata=[]
    #remove the slice operation to get the complete dateset
    for ai, article in enumerate(tqdm(source_data['data'][0:1])):
        #remove the slice operation to get the complete dataset
        for pi, para in enumerate(article['paragraphs'][0:1]):
            context=para['context'].replace("''", '" ').replace("``", '" ')
            xi=list(map(word_tokenize,nltk.sent_tokenize(context)))
            xi=[process_tokens(tokens) for tokens in xi]
            for sen in xi:
                for word in sen:
                    word_counter[word]+=len(para['qas'])
                    
            for qa in para['qas']:
                q=qa['question']
                qi=word_tokenize(qa['question'])
                answers=[]
                for ans in qa['answers']:
                    answer_text=ans['text']
                    #answer=word_tokenize(answer_text)
                    answers.append(answer_text)
                for word in qi:
                    word_counter[word]+=1
                retdata.append([q,answers,context])
    #print(retdata[-1])
    return retdata,word_counter

def word_tokenize(tokens):
    #tokens is a string(sentence)
    return [token.replace("''",'"').replace('``','"') for token in nltk.word_tokenize(tokens)]

def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens

class Args():
    lib_dir='/home/shr/NLP/treelstm/lib'
    classpath=':'.join([
        lib_dir,
        os.path.join(lib_dir,'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

def constituency_parse(sentence,cp='',tokenize=True):
    args=Args()
    with open('tmp.txt','w+') as outfile:
        outfile.write(sentence)
    tokpath='tmp.tok'
    parentpath='tmp.cparents'
    relpath='tmp.rel'
    #cmd=('java -cp {} DependencyParse -tokpath {} -parentpath {} -relpath {} -tokenize -  < {}'.format(args.classpath,tokpath,parentpath, relpath,'tmp.txt'))
    #os.system(cmd)
    cmd=('java -cp {} ConstituencyParse -tokpath {} -parentpath {} -tokenize -  < {}'.format(args.classpath,tokpath,parentpath,'tmp.txt'))
    os.system(cmd)
def parse_tree(sentence,parents):
    nodes = {}
    parents = [p - 1 for p in parents]  #change to zero based
    sentence=[w for w in sentence.strip().split()]
    for i in range(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = tNode(idx) 
                if prev is not None:
                    assert prev.idx != node.idx
                    node.add_child(prev)
                nodes[idx] = node
                if idx < len(sentence):
                    node.word = sentence[idx]
                parent = parents[idx]
                if parent in nodes:
                    assert len(nodes[parent].children) < 2
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break
                prev = node
                idx = parent
    return root

def load_tree(tokfile,parentsfile):
    sentence=[]
    parents=[]
    with open(tokfile) as infile:
        lines=infile.readlines()
        assert len(lines)==1
        sentence=lines[0]
    with open(parentsfile) as infile:
        lines=infile.readlines()
        assert len(lines)==1
        parents=lines[0].strip().split()
    parents=[int(parent) for parent in parents]
    return parse_tree(sentence,parents)

def get_tree(sentence):#由一个句子获得一棵树
    constituency_parse(sentence)
    root=load_tree('tmp.tok','tmp.cparents')
    postOrder=root.postOrder
    postOrder(root,tNode.get_height,None) 
    postOrder(root,tNode.get_numleaves,None) 
    postOrder(root,root.get_spans,None)
    postOrder(root,root.print_span,None)
    print(root.height,root.num_leaves)
    return root

def extract_filled_tree(cur_data,fillnum=200,word2idx=None):
    #cur_data is a treeroot
    dim2=fillnum
    #dim1: batch_size
    #dim2: tree node size
    leaf_emb_arr = np.empty([dim2],dtype='int32')
    leaf_emb_arr.fill(-1)
    treestr_arr = np.empty([dim2,2],dtype='int32')
    treestr_arr.fill(-1)
    t_leaf_emb_arr =np.empty([dim2],dtype='int32')
    t_leaf_emb_arr.fill(-1)
    t_treestr_arr=np.empty([dim2],dtype='int32')
    t_treestr_arr.fill(-1)
    t_par_leaf_arr=np.empty([dim2],dtype='int32')
    t_par_leaf_arr.fill(-1)
    tree=cur_data
    input_, treestr, t_input,t_treestr,t_par_leaf=extract_tree_data(tree,max_degree=2, only_leaves_have_vals=False,word2idx=word2idx)
    leaf_emb_arr[0:len(input_)]=input_
    treestr_arr[0:len(treestr),0:2]=treestr
    t_leaf_emb_arr[0:len(t_input)]=t_input
    t_treestr_arr[0:len(t_treestr)]=t_treestr
    t_par_leaf_arr[0:len(t_par_leaf)]=t_par_leaf
    return leaf_emb_arr, treestr_arr,t_leaf_emb_arr,t_treestr_arr,t_par_leaf_arr

def extract_tree_data(tree, word2idx=None,max_degree=2, only_leaves_have_vals=True):
    leaves, inodes=BFStree(tree,word2idx)
    leaf_emb=[]
    tree_str=[]
    t_leaf_emb=[]
    t_tree_str=[]
    i=0
    for leaf in reversed(leaves):
        leaf.idx=i
        i+=1
        leaf_emb.append(leaf.word)
    for node in reversed(inodes):
        node.idx=i
        c=[child.idx for child in node.children]
        tree_str.append(c)
        i+=1
    i=0
    for node in inodes:
        node.tidx=i
        i+=1
        if node.parent:
            t_tree_str.append(node.parent.tidx)
        else:
            t_tree_str.append(-1)
    t_par_leaf=[]
    for leaf in leaves:
        leaf.tidx=i
        i+=1
        t_par_leaf.append(leaf.parent.tidx)
        t_leaf_emb.append(leaf.word)
    print('{}:leaf'.format(leaf_emb))
    print('{}.tree'.format(tree_str))
    print('{}.t_tree'.format(t_tree_str))
    print('{}.t_leaf'.format(t_leaf_emb))
    print('{}.t_par_leaf'.format(t_par_leaf))
    return (np.array(leaf_emb,dtype='int32'),np.array(tree_str,dtype='int32'),np.array(t_leaf_emb,dtype='int32'),np.array(t_tree_str,dtype='int32'),np.array(t_par_leaf,dtype='int32'))

def BFStree(root, word2idx=None):
    from collections import deque
    node=root
    leaves=[]
    inodes=[]
    queue=deque([node])
    func=lambda node:node.children==[]
    while queue:
        node=queue.popleft()
        if func(node):
            print(node.word)
            if word2idx:
                if word2idx.get(node.word):
                    node.word=word2idx[node.word]
                elif word2idx.get(node.word.lower()):
                    node.word=word2idx[node.word.lower()]
                else:
                    logging.warn('no word2idx for question/context {}'.format(node.word))
                    node.word=word2idx['unknown']
            leaves.append(node)
        else:
            inodes.append(node)
        if node.children:
            for child in node.children:
                child.add_parent(node)
            queue.extend(node.children)
    return leaves,inodes

def get_max_len_data(data):
    train_data=data['train']
    dev_data=data['dev']

def candidate_answer_generate(answer_data, context_sentence_roots_list):
    #candidate_answers: sentence_num * candidate_number * constituency_num, each is a constituency id list(reversed BFS order)
    #correct_answer_idx
    candidate_answers=[]
    correct_answer_idx=-1
    candidate_answer_overall_number=0
    sentence_num=len(context_sentence_roots_list)
    overall_idx=-1
    for root in context_sentence_roots_list:
        overall_idx+=1
        cur_candidate_answer=[]
        constituency_id2span={}
        leaf_num=0
        queue=deque([node])
        while queue:
            node=queue.poplest()
            if node.children!=[]:
                candidate_answer_overall_number+=1
                cur_candidate_answer.append([node.idx])
                overall_idx+=1
                queue.extend(node.children)
                constituency_id2span[node.idx]=node.span
                if node.get_spans==answer_data:
                    if correct_answer_idx!=-1:
                        logging.warn('{} has duplicated candidate answers'.format(root.span))
                        correct_answer_idx=overall_idx
                    else:
                        correct_answer_idx=overall_idx

        candidate_answers.append(cur_candidate_answer)

    return candidate_answers, correct_answer_idx, candidate_answer_overall_number

if __name__ =='__main__':                                                                                                                       
    root=get_tree('Yet the act is still charming here.')
    word2idx,embedding=load_embedding()
    #leaves,inodes=BFStree(root)
    #for node in inodes:
    #    print(len(node.children))
    b_input, b_treestr, t_input, t_treestr, t_parent=extract_filled_tree(root,word2idx=word2idx)
    print(b_input)
    print(b_treestr)
    print(t_input)
    print(t_treestr)
    print(t_parent)

# A Constituent-Centric Neural Architecture for Reading Comprehension

Implemented in Tensorflow, python3.6
Not finished, but open to be revised.

@Author Haoran Shi

Files:
- Core Model File: ``ccrc_model.py``
- Sub-modules:
    - question encoding: ``question_encoding.py``
    - context encoding: ``context_encoding.py``
    - attention layer:``attention_layer.py``
    - data utilities and candidate answers generation:``load_data.py``
    - answer prediction and parameter learning:``ccrc_model.py``

I have implemented all the modules of the model described in *A constituent-Centric Neural Architecture for Reading Comprehension*, except that some details in candidate answer generation module has not been finished, and the ``load_data.py`` can be optimized by means of generating some intermediate files, instead of reading from original SQuAD dataset.

When building the bottom-up lstm tree of question, I refer to some basic data processing and fundemental architecture of https://github.com/stanfordnlp/treelstm and  https://github.com/shrshore/RecursiveNN

Every module has a ``main`` method and you can run it to check the result.

For debugging, I set some constraints in ``load_data.py``, you can see related information at the first line of the module.

## Prerequisites
1. python3.6
2. tensorflow 1.0.1
## Pipeline
0. git clone https://github.com/shrshore/Constituent-Centric-Neural-Architecture-for-Reading-Comprehension
1. git clone https://github.com/allenai/bi-att-flow, enter the directory and run the ``download.sh``, you will get the glove vectors and squad dataset.
2. git clone https://github.com/stanfordnlp/treelstm and enter the directory
3. run ``./fetch_and_preprocess.sh`` to download **glove** word vectors and **stanford parser**.
4. download [squad](https://rajpurkar.github.io/SQuAD-explorer/) into the same directory of glove The filepath can be modified in ``load_data.py``
5. enter the directory of *Constituent-Centric-Neural-Architecture-for-Reading-Comprehension*
6. run ``my_main.py``. 
7. You can check the hidden value representation of root node for every question in ``logger.log`` file

## Comments

Actually there are something confusing me in the original paper in question-encoding part. 

1. Near line 269, it says that *for internal node except for root, the inputs are the hidden states $h_{\downarrow}^{(p)}$ and memory cell $c_{\downarrow}^{(p)}$ of its parents*. Actually I think the *parents* should be *parent* because in the binary constituency tree, one node only has one parent except for the root node.
2. For the root node, the $h_{\downarrow}^{(r)}$ is set to $h_{\uparrow}^{(r)}$, but I'm not sure what the $c_{\downarrow}^{r}$ of root node should be, which is the input of its children. So I just set $c_{\downarrow}^{r}$ to $c_{\uparrow}^{(r)}$.

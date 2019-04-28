from collections import defaultdict
import numpy as np

LABEL2VEC = {
    'news': [1, 0, 0, 0],
    'non-rumor': [1, 0, 0, 0],
    'false': [0, 1, 0, 0],
    'true': [0, 0, 1, 0],
    'unverified': [0, 0, 0, 1],
}

class Node(object):
    def __init__(self, idx=None, parent=None):
        self.children = []
        self.idx = idx
        self.parent = None
        self.word = [] # actually word freq
        self.word_idx = []

def loadData(label_file, tree_file, train_file, test_file, logger=print):
    # 1. loading tree label
    logger('loading tree label')
    label_dict = defaultdict(dict)
    for line in open(label_file):
        label, _, eid, _, _, _, _, _, _ = line.rstrip().split('\t')
        label_dict[eid] = label.lower()
    print('done. load {} labels'.format(len(label_dict)))
    
    # 2. loading tree nodes data
    logger('loading tree ...')
    tree_dict = defaultdict(dict)
    for line in open(tree_file):
        eid, parent_idx, node_idx, num_parent, text_len, vec = line.rstrip().split('\t')
        tree_dict[eid][int(node_idx)] = {
            'parent': parent_idx, 
            'num_parent': int(num_parent), 
            'text_len': int(text_len), 
            'vec': vec
        }
    logger('done. load {} trees'.format(len(tree_dict)))

    # 3. loading train set
    logger('loading train set and construct tree ...')
    label_count = defaultdict(lambda: 0)
    tree_train, word_train, index_train, y_train, parent_num_train = [], [], [], [], []
    for line in open(train_file):
        eid = line.rstrip()
        # 3.1 load label
        # print('loading label and construct tree of eid({}) ...'.format(eid))
        if eid not in label_dict or eid not in tree_dict: continue
        if len(tree_dict[eid]) <= 0: continue
        label = label_dict[eid]
        y_train.append(LABEL2VEC[label])
        label_count[label] += 1
        # 3.2 construct tree
        x_word, x_index, tree, parent_num = constructTree(tree_dict[eid])
        tree_train.append(tree)
        word_train.append(x_word)
        index_train.append(x_index)
        parent_num_train.append(parent_num)
        # print('done.')
    print('done. in train set number of all class label: {}'.format(dict(label_count)))

    # 4. loading test set
    logger('loading test set and construct tree ...')
    label_count = defaultdict(lambda: 0)
    tree_test, word_test, index_test, y_test, parent_num_test = [], [], [], [], []
    for line in open(test_file):
        eid = line.rstrip()
        # 4.1 load label
        # print('loading label and construct tree of eid({}) ...'.format(eid))
        if eid not in label_dict or eid not in tree_dict: continue
        if len(tree_dict[eid]) <= 0: continue
        label = label_dict[eid]
        y_test.append(LABEL2VEC[label])
        label_count[label] += 1
        # 4.2 construct tree
        x_word, x_index, tree, parent_num = constructTree(tree_dict[eid])
        tree_test.append(tree)
        word_test.append(x_word)
        index_test.append(x_index)
        parent_num_test.append(parent_num)
        # print('done.')
    print('done. in test set number of all class label: {}'.format(dict(label_count)))

    return tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test

def str2matrix(string, text_len): # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    l = 0
    for pair in string.split(' '):
        wordFreq.append(float(pair.split(':')[1]))
        wordIndex.append(int(pair.split(':')[0]))
        l += 1
    ladd = [ 0 for i in range( text_len-l ) ]
    wordFreq += ladd
    wordIndex += ladd
    return wordFreq, wordIndex

def constructTree(node_dict):
    index2node = {}
    for node_idx in node_dict:
        index2node[node_idx] = Node(idx=node_idx)

    # print('connect parents and children ...')
    for node_idx in node_dict:
        parent_idx = node_dict[node_idx]['parent']
        node = index2node[node_idx]
        word_freq, word_idx = str2matrix(node_dict[node_idx]['vec'], node_dict[node_idx]['text_len'])

        node.word = word_freq
        node.word_idx = word_idx

        if parent_idx == 'None':
            root = node
        else:
            node_parent = index2node[int(parent_idx)]
            node.parent = node_parent
            node_parent.children.append(node)

    num_parent = node_dict[node_idx]['num_parent']
    ini_x, ini_index = str2matrix("0:0", node_dict[node_idx]['text_len'])
    x_word, x_index, tree = gen_nn_inputs(root, ini_x)
    return x_word, x_index, tree, num_parent

def gen_nn_inputs(root_node, ini_word):
    tree = [[0, root_node.idx]]
    X_word, X_index = [root_node.word], [root_node.word_idx]

    if not root_node.children:
        internal_tree = []
        internal_word = []
        internal_index = []
    layers = []
    layer = [root_node]
    # get every layer's nodes
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child]) for node in layer]
        layer = next_layer
    internal_tree = []
    internal_word = []
    internal_index = []
    for layer in layers:
        for node in layer:
            if not node.children:
                continue
            for child in node.children:
                internal_tree.append([node.idx, child.idx])
                internal_word.append(child.word if child.word is not None else -1)
                internal_index.append(child.word_idx if child.word_idx is not None else -1)

    tree.extend(internal_tree)
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    X_word.append(ini_word)

    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'))

if __name__ == '__main__':
    tree_file = '/home/einstein/Workspace/barn/Rumor_RvNN/resource/data.TD_RvNN.vol_5000.txt'
    label_file = '/home/einstein/Workspace/barn/Rumor_RvNN/resource/Twitter15_label_All.txt'
    train_file = '/home/einstein/Workspace/barn/Rumor_RvNN/nfold/RNNtrainSet_Twitter152_tree.txt'
    test_file = '/home/einstein/Workspace/barn/Rumor_RvNN/nfold/RNNtestSet_Twitter152_tree.txt'
    loadData(label_file, tree_file, train_file, test_file)

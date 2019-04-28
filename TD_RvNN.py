__doc__ = """Tree GRU aka Recursive Neural Networks."""

import numpy as np

obj = "Twitter15" # choose dataset, you can choose either "Twitter15" or "Twitter16"
fold = "2" # fold index, choose from 0-4
tag = "_u2b"
vocabulary_size = 5000
hidden_dim = 100
Nclass = 4
Nepoch = 600
lr = 0.005
unit="TD_RvNN-"+obj+str(fold)+'-vol.'+str(vocabulary_size)+tag

treePath = 'resource/data.TD_RvNN.vol_'+str(vocabulary_size)+'.txt'

trainPath = "nfold/RNNtrainSet_"+obj+str(fold)+"_tree.txt"
testPath = "nfold/RNNtestSet_"+obj+str(fold)+"_tree.txt"
labelPath = "resource/"+obj+"_label_All.txt"


class Node_tweet(object):
	def __init__(self, idx=None):
		self.children = []
		# self.index = index
		self.idx = idx
		self.word = []
		self.index = []
		# self.height = 1
		# self.size = 1
		# self.num_leaves = 1
		self.parent = None
	# self.label = None

################################## tools ########################################33
def str2matrix(Str, MaxL):  # str = index:wordfreq index:wordfreq
	wordFreq, wordIndex = [], []
	l = 0
	for pair in Str.split(' '):
		wordFreq.append(float(pair.split(':')[1]))
		wordIndex.append(int(pair.split(':')[0]))
		l += 1
	ladd = [0 for i in range(MaxL - l)]
	wordFreq += ladd
	wordIndex += ladd
	# print MaxL, l, len(Str.split(' ')), len(wordFreq)
	# print Str.split(' ')
	return wordFreq, wordIndex


def loadLabel(label, l1, l2, l3, l4):
	labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
	if label in labelset_nonR:
		y_train = [1, 0, 0, 0]
		l1 += 1
	if label in labelset_f:
		y_train = [0, 1, 0, 0]
		l2 += 1
	if label in labelset_t:
		y_train = [0, 0, 1, 0]
		l3 += 1
	if label in labelset_u:
		y_train = [0, 0, 0, 1]
		l4 += 1
	return y_train, l1, l2, l3, l4


def constructTree(tree):
	## tree: {index1:{'parent':, 'maxL':, 'vec':}
	## 1. ini tree node
	index2node = {}
	for i in tree:
		node = Node_tweet(idx=i)
		index2node[i] = node
	## 2. construct tree
	for j in tree:
		indexC = j
		indexP = tree[j]['parent']
		nodeC = index2node[indexC]
		wordFreq, wordIndex = str2matrix(tree[j]['vec'], tree[j]['maxL'])
		# print tree[j]['maxL']
		nodeC.index = wordIndex
		nodeC.word = wordFreq
		# nodeC.time = tree[j]['post_t']
		## not root node ##
		if not indexP == 'None':
			nodeP = index2node[int(indexP)]
			nodeC.parent = nodeP
			nodeP.children.append(nodeC)
		## root node ##
		else:
			root = nodeC
	## 3. convert tree to DNN input
	parent_num = tree[j]['parent_num']
	ini_x, ini_index = str2matrix("0:0", tree[j]['maxL'])
	# x_word, x_index, tree = tree_gru_u2b.gen_nn_inputs(root, ini_x, ini_index)
	x_word, x_index, tree = gen_nn_inputs(root, ini_x)
	return x_word, x_index, tree, parent_num


################################# loas data ###################################
def loadData():
	print("loading tree label")
	labelDic = {}
	for line in open(labelPath):
		line = line.rstrip()
		label, eid = line.split('\t')[0], line.split('\t')[2]
		labelDic[eid] = label.lower()
	print(len(labelDic))

	print("reading tree")  ## X
	treeDic = {}
	for line in open(treePath):
		line = line.rstrip()
		eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
		parent_num, maxL = int(line.split('\t')[3]), int(line.split('\t')[4])
		Vec = line.split('\t')[5]
		if not eid in treeDic:
			treeDic[eid] = {}
		treeDic[eid][indexC] = {'parent': indexP, 'parent_num': parent_num, 'maxL': maxL, 'vec': Vec}
	print('tree no:', len(treeDic))

	print("loading train set")
	tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
	l1, l2, l3, l4 = 0, 0, 0, 0
	for eid in open(trainPath):
		# if c > 8: break
		eid = eid.rstrip()
		if not eid in labelDic: continue
		if not eid in treeDic: continue
		if len(treeDic[eid]) <= 0:
			# print labelDic[eid]
			continue
		## 1. load label
		label = labelDic[eid]
		y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
		y_train.append(y)
		## 2. construct tree
		# print eid
		x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
		tree_train.append(tree)
		word_train.append(x_word)
		index_train.append(x_index)
		parent_num_train.append(parent_num)
		# print treeDic[eid]
		# print tree, child_num
		# exit(0)
		c += 1
	print(l1, l2, l3, l4)

	print("loading test set")
	tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
	l1, l2, l3, l4 = 0, 0, 0, 0
	for eid in open(testPath):
		# if c > 4: break
		eid = eid.rstrip()
		if not eid in labelDic: continue
		if not eid in treeDic: continue
		if len(treeDic[eid]) <= 0:
			# print labelDic[eid]
			continue
		## 1. load label
		label = labelDic[eid]
		y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
		y_test.append(y)
		## 2. construct tree
		x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
		tree_test.append(tree)
		word_test.append(x_word)
		index_test.append(x_index)
		parent_num_test.append(parent_num)
		c += 1
	print(l1, l2, l3, l4)
	print("train no:", len(tree_train), len(word_train), len(index_train), len(parent_num_train), len(y_train))
	print("test no:", len(tree_test), len(word_test), len(index_test), len(parent_num_test), len(y_test))
	print("dim1 for 0:", len(tree_train[0]), len(word_train[0]), len(index_train[0]))
	print("case 0:", tree_train[0][0], word_train[0][0], index_train[0][0], parent_num_train[0])
	# print index_train[0]
	# print word_train[0]
	# print tree_train[0]
	# exit(0)
	return tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test


################################# generate tree structure ##############################
# def gen_nn_inputs(root_node, ini_word, ini_index):
def gen_nn_inputs(root_node, ini_word):
	tree = [[0, root_node.idx]]
	X_word, X_index = [root_node.word], [root_node.index]

	internal_tree, internal_word, internal_index = _get_tree_path(root_node)
	tree.extend(internal_tree)
	X_word.extend(internal_word)
	X_index.extend(internal_index)
	X_word.append(ini_word)
	return (np.array(X_word, dtype='float32'),
	        np.array(X_index, dtype='int32'),
	        np.array(tree, dtype='int32'))

def _get_tree_path(root_node):
	if not root_node.children:
		return [], [], []
	layers = []
	layer = [root_node]
	while layer:
		layers.append(layer[:])
		next_layer = []
		[next_layer.extend([child for child in node.children if child])
		 for node in layer]
		layer = next_layer
	tree = []
	word = []
	index = []
	for layer in layers:
		for node in layer:
			if not node.children:
				continue
			for child in node.children:
				tree.append([node.idx, child.idx])
				word.append(child.word if child.word is not None else -1)
				index.append(child.index if child.index is not None else -1)

	return tree, word, index
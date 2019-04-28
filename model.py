import torch, torchvision
from torch import nn
import numpy as np

device = 'cuda'


def _getTreeInfo(tree):
	idx_dict = {}
	for _t in tree:
		parent_idx = _t[0]
		child_idx = _t[1]
		idx_dict[parent_idx] = 'not leaf'
		idx_dict[child_idx] = 'leaf'
	leaves_idx = [_i for _i in idx_dict if idx_dict[_i] == 'leaf']
	return {'num_node': len(idx_dict), 'num_leaf': len(leaves_idx), 'leaves_idx': leaves_idx}

class RvNN(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, num_class):
		super(RvNN, self).__init__()
		assert vocab_size > 1 and hidden_size > 1, 'vocab_size or hidden_sizem <= 1 !'
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_class = num_class

		# self.E = torch.zeros(self.hidden_size, self.vocab_size, requires_grad=True).to(device)+0.01
		# self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
		# self.word_fc = nn.Linear(in_features=vocab_size, out_features=hidden_size)
		# self.E = nn.Parameter(torch.zeros(self.hidden_size, self.vocab_size, requires_grad=True).to(device) + 0.01)
		self.E = nn.Parameter(torch.normal(mean=torch.zeros(self.hidden_size, self.vocab_size, requires_grad=True), std=0.1))
		self.gru = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
		self.gru = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
		self.output_fc = nn.Linear(in_features=hidden_size, out_features=num_class)
		self.softmax = nn.Softmax(dim=0)

		self.gru.weight_hh.data.normal_(0, 0.1)
		self.gru.weight_ih.data.normal_(0, 0.1)
		self.output_fc.weight.data.normal_(0, 0.1)

		self.gru.bias_hh.data.fill_(0)
		self.gru.bias_ih.data.fill_(0)
		self.output_fc.bias.data.fill_(0)

		# self.gru.weight_hh.data.fill_(0.2)
		# self.gru.weight_ih.data.fill_(0.2)
		# self.output_fc.weight.data.fill_(0.2)
		#
		# self.gru.bias_hh.data.fill_(0.1)
		# self.gru.bias_ih.data.fill_(0.1)
		# self.output_fc.bias.data.fill_(0.1)

	def initEmbedding(self):
		# init with glove, current remain empty
		pass

	def initHiddenState(self):
		return np.zeros([self.hidden_size], dtype=np.float32)


	def forward(self, tree, word, word_idx):
		'''
		:param tree_root:   type(Node)
		:param h_parent:    type(torch.tensor), (1,hidden_size)
		:return:
		'''
		tree_info = _getTreeInfo(tree)
		h_list = torch.zeros(tree_info['num_node'], self.hidden_size, requires_grad=True).to(device)
		for i in range(len(tree)):
			parent_idx = tree[i][0]
			child_idx = tree[i][1]
			parent_h = h_list[parent_idx]
			word_idx_tensor = torch.LongTensor(word_idx[i]).to(device)
			word_tensor = torch.FloatTensor(word[i]).unsqueeze(dim=1).to(device)
			E_child = self.E[:, word_idx_tensor]
			child_xe = E_child.mm(word_tensor).squeeze(dim=1)

			batch_input = child_xe.unsqueeze(dim=0)
			batch_hidden = parent_h.unsqueeze(dim=0)
			batch_child_h = self.gru(batch_input, batch_hidden)

			child_h = batch_child_h.squeeze(dim=0)
			h_list = torch.cat((h_list[:child_idx, :], child_h.unsqueeze(dim=0), h_list[child_idx+1:, :]), dim=0).to(device)
			# h_list.requires_grad = True

		children_h = h_list[tree_info['leaves_idx']]
		# children_h = h_list
		final_state = children_h.max(dim=0)[0]
		output = self.output_fc(final_state)
		# pred = output
		pred = self.softmax(output)
		return pred

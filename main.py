
import misc, model, eval, evaluate

import numpy as np
import torch, time, random
from torch import nn

import TD_RvNN

device = 'cuda'

tree_file = 'resource/data.TD_RvNN.vol_5000.txt'
label_file = 'resource/Twitter15_label_All.txt'
train_file = 'nfold/RNNtrainSet_Twitter152_tree.txt'
test_file = 'nfold/RNNtestSet_Twitter152_tree.txt'

vocab_size = 5000
embed_size = 512
hidden_size = 100
num_class = 4
epoches = 600
lr = 0.005
# lr = 1

# tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test = TD_RvNN.loadData(label_file, tree_file, train_file, test_file)

tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test = TD_RvNN.loadData()
# print("train no:", len(tree_train), len(word_train), len(index_train),len(parent_num_train), len(y_train))
# print("test no:", len(tree_test), len(word_test), len(index_test), len(parent_num_test), len(y_test))
# print("dim1 for 0:", len(tree_train[0]), len(word_train[0]), len(index_train[0]))
# print("case 0:", tree_train[0][0], word_train[0][0], index_train[0][0], parent_num_train[0])

model = model.RvNN(
	vocab_size=vocab_size,
	embed_size=embed_size,
	hidden_size=hidden_size,
	num_class=num_class
).to(device)

loss_func = nn.MSELoss(reduction='sum')

model_optimizer = torch.optim.SGD(
	# params=filter(lambda p: p.requires_grad, model.parameters()),
	params=model.parameters(),
	momentum=0.9,
	lr=lr
)

losses_5, losses = [], []
num_examples_seen = 0
for epoch in range(epoches):
	t_s = time.time()
	train_idx_list = [_i for _i in range(len(y_train))]
	# random.shuffle(train_idx_list)
	for train_idx in train_idx_list:
		# pred = model.forward(tree_train[train_idx+1], word_train[train_idx+1], index_train[train_idx+1])
		pred = model.forward(tree_train[train_idx], word_train[train_idx], index_train[train_idx])
		target = torch.FloatTensor(y_train[train_idx]).to(device)
		loss = loss_func(pred, target)

		model_optimizer.zero_grad()
		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
		model_optimizer.step()

		losses.append(loss.data.cpu().numpy())
		num_examples_seen += 1

	print('epoch={}: loss={:.6f}, takes {:.2f}s'.format(epoch, np.mean(losses), time.time()-t_s))

	if epoch % 5 == 0:
		losses_5.append((num_examples_seen, np.mean(losses)))
		print('Loss after num_examples_seen={}, epoch={}, loss={:.6f}'.format(num_examples_seen, epoch, np.mean(losses)))

		# enter prediction
		prediction = []
		for test_idx in range(len(y_test)):
			pred = model.forward(tree_train[test_idx], word_train[test_idx], index_train[test_idx])
			prediction.append(pred.unsqueeze(dim=0).cpu().data.numpy())
		res = evaluate.evaluation_4class(prediction, y_test)

		print('results: {}'.format(res))

		if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
			lr = lr * 0.5
			print("Setting learning rate to {}".format(lr))

			model_optimizer = torch.optim.SGD(
				params=model.parameters(),
				momentum=0.9,
				lr=lr
			)

	losses = []
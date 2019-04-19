import numpy as np
import part1_data
import math
from copy import copy
import pudb

def prep_data():
	res, docs_sparse = part1_data.get_data()
	mini_batches = part1_data.data_loader(docs_sparse, res, 1)
	mini_batch_train = mini_batches[:int(0.8 * len(mini_batches))]
	mini_batch_inf = mini_batches[int(0.8 * len(mini_batches)):]

	return mini_batch_train, mini_batch_inf

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return (sigmoid(x) - 1) / sigmoid(x)

def relu_derivative(x):
	return (x > 0)
	
def CrossEntropy(Y_pred, y):
	# pu.db
	# Y_pred += 1e-6
	if y == 1:
		return -np.log(Y_pred)
	else:
		return -np.log((1 - Y_pred) + 1e-6)
		
def forward(X, W1, W2):
	hid = np.dot(np.reshape(X, (500)), W1)
	hid = hid * (hid > 0)
	y_pred = np.dot(hid,W2)
	# pu.db
	return hid, sigmoid(y_pred[0])

def train(batch, epochs):
	hid_layer = np.random.randn(100)
	W1 = np.random.randn(500, 100)
	W2 = np.random.randn(100, 1)
	net_loss = 0.0

	def back(X, Y_pred, Y, hidden, W1, W2, epoch):
		# print("Here")
		del_W2 = np.zeros((100))
		for i in range(len(Y)):
			del_W2 = del_W2 + np.dot(hidden[i].T, (((Y_pred[i] - Y[i])/ (Y_pred[i] - Y_pred[i]**2 + 1e-6)) * sigmoid_derivative(Y[i])))

		del_W1 = np.zeros((500, 100))
		for i in range(len(Y)):
			try:
				del_W1 = del_W1 + np.dot(X[i],  (np.dot(((Y_pred[i] - Y[i])/ (Y_pred[i] - Y_pred[i]**2 + 1e-6)) * sigmoid_derivative(Y[i]), W2.T) * relu_derivative(hidden[i])))
			except:
				pu.db

		# pu.db
		del_W2 = np.reshape(del_W2, (100, 1))
		# try:
		W2 = W2 - 0.1 * (del_W2 / len(Y))
		W1 = W1 - 0.1 * (del_W1 / len(Y))

		return W1, W2
		if (epoch == 3):
			pu.db
		# try:
		# 	pass
		# except:
		# 	pu.db

	for epoch in range(epochs):
		# input()
		print("New epoch: "+str(epoch))
		net_loss = 0.0
		for each in batch:
			X = each[0]
			Y = each[1]
			Y_pred = []
			hidden = []
			loss = 0.0
			for i in range(len(X)):
				# pu.db
				X_here = X[i]
				Y_here = Y[i]
				hid_layer, Y_pred_here = forward(X_here, W1, W2)
				loss += CrossEntropy(Y_pred_here, Y_here)
				Y_pred.append(Y_pred_here)
				hidden.append(hid_layer)

			net_loss += loss / len(X)
			# W2_copy = copy(W2)
			W1, W2 = back(X, Y_pred, Y, hidden, W1, W2, epoch)
			# pu.db
		print("loss: ", net_loss)

if __name__ == "__main__":
	train_batch, test_batch = prep_data()
	train(train_batch, 100)
	# pu.db

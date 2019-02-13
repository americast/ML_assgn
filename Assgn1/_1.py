import numpy as np
import random
import matplotlib.pyplot as plt

def train(NUM_DATA =10, MAX_ITER = 5000, ALPHA = 0.05):

	X = np.random.uniform(0, 1, NUM_DATA)


	noise = np.random.normal(0, 0.3, NUM_DATA)

	Y = np.sin(2 * np.pi * X) + noise

	cutoff = int(0.8 * NUM_DATA)

	X_test = X[cutoff:]
	Y_test = Y[cutoff:]

	X_train = X[:cutoff]
	Y_train = Y[:cutoff]

	np.save("X_train", X_train)
	np.save("X_test", X_test)
	np.save("Y_train", Y_train)
	np.save("Y_test", Y_test)

	train_error = []
	test_error = []

	for DEGREE in range(1, 10):
		print("\nDegree: "+str(DEGREE))
		W = np.random.uniform(0, 1, DEGREE + 1)

		for i in range(MAX_ITER):

			cost = 0.0
			cost_val = 0.0
			sum_ = 0.0
			sum_list = np.zeros(DEGREE + 1)
			for m in range(len(X_train)):
				X = X_train[m]
				Y = Y_train[m]
				X_power = np.array([X**j for j in range(DEGREE+1)])

				Y_pred = np.dot(W, X_power)
				sum_ = (Y_pred - Y)
				sum_list_here = [sum_ * j for j in X_power]
				sum_list += sum_list_here

				cost += (Y - Y_pred)**2

			cost/= (2 * (m+1))

			W -= (ALPHA/(len(X_train))) * sum_list
			print("Cost at iter "+str(i)+"/"+str(MAX_ITER)+": "+str(cost), end="\r")

			for m in range(len(X_test)):
				X = X_test[m]
				Y = Y_test[m]
				X_power = np.array([X**j for j in range(DEGREE+1)])

				Y_pred = np.dot(W, X_power)

				cost_val += (Y - Y_pred)**2

			cost_val /= 2* (m+1)

		train_error.append(cost)
		test_error.append(cost_val)
		np.save("W_"+str(DEGREE), W)

	return train_error, test_error

if __name__ == "__main__":
	train_error, test_error = train()
	print(train_error)
	print(test_error)
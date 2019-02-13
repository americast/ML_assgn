import _1
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def train_abs(NUM_DATA =10, MAX_ITER = 5000, ALPHA = 0.05):

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
	rmse_error = []

	for DEGREE in range(1, 10):
		print("\nDegree: "+str(DEGREE))
		W = np.random.uniform(0, 1, DEGREE + 1)

		for i in range(MAX_ITER):

			cost = 0.0
			cost_val = 0.0
			sum_ = 0.0
			rmse = 0.0
			sum_list = np.zeros(DEGREE + 1)
			for m in range(len(X_train)):
				X = X_train[m]
				Y = Y_train[m]
				X_power = np.array([X**j for j in range(DEGREE+1)])

				Y_pred = np.dot(W, X_power)
				sum_ = (Y_pred - Y)/math.fabs(Y_pred - Y)
				sum_list_here = [sum_ * j for j in X_power]
				sum_list += sum_list_here

				cost += math.fabs(Y - Y_pred)


			cost/= (2 * (m+1))

			W -= (ALPHA/(2 *len(X_train))) * sum_list
			print("Cost at iter "+str(i)+"/"+str(MAX_ITER)+": "+str(cost), end="\r")

			for m in range(len(X_test)):
				X = X_test[m]
				Y = Y_test[m]
				X_power = np.array([X**j for j in range(DEGREE+1)])

				Y_pred = np.dot(W, X_power)

				cost_val += math.fabs(Y - Y_pred)
				rmse += (Y - Y_pred)**2

			cost_val /= 2* (m+1)
			rmse = math.sqrt(rmse)

		train_error.append(cost)
		test_error.append(cost_val)
		rmse_error.append(rmse)
		np.save("W_"+str(DEGREE), W)

	return train_error, test_error, rmse_error


def train_fourth(NUM_DATA =10, MAX_ITER = 5000, ALPHA = 0.05):

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
	rmse_error = []

	for DEGREE in range(1, 10):
		print("\nDegree: "+str(DEGREE))
		W = np.random.uniform(0, 1, DEGREE + 1)

		for i in range(MAX_ITER):

			cost = 0.0
			cost_val = 0.0
			sum_ = 0.0
			rmse = 0.0
			sum_list = np.zeros(DEGREE + 1)
			for m in range(len(X_train)):
				X = X_train[m]
				Y = Y_train[m]
				X_power = np.array([X**j for j in range(DEGREE+1)])

				Y_pred = np.dot(W, X_power)
				sum_ = (Y_pred - Y)**3
				sum_list_here = [sum_ * j for j in X_power]
				sum_list += sum_list_here

				cost += (Y - Y_pred)**4

			cost/= (2 * (m+1))

			W -= (2 * ALPHA/(len(X_train))) * sum_list
			print("Cost at iter "+str(i)+"/"+str(MAX_ITER)+": "+str(cost), end="\r")

			for m in range(len(X_test)):
				X = X_test[m]
				Y = Y_test[m]
				X_power = np.array([X**j for j in range(DEGREE+1)])

				Y_pred = np.dot(W, X_power)

				cost_val += (Y - Y_pred)**4
				rmse += (Y - Y_pred)**2

			cost_val /= 2* (m+1)
			rmse = math.sqrt(rmse)


		train_error.append(cost)
		test_error.append(cost_val)
		rmse_error.append(rmse)
		np.save("W_"+str(DEGREE), W)

	return train_error, test_error, rmse_error


if __name__ == "__main__":

	# For RMSE of squared error
	X_test_rmse = []
	X_axis = [0.025, 0.05, 0.1, 0.2, 0.5]

	train_error, test_error, rmse_error = _1.train(100, 500, 0.025)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = _1.train(100, 500, 0.05)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = _1.train(100, 500, 0.1)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = _1.train(100, 500, 0.2)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = _1.train(100, 500, 0.5)
	X_test_rmse.append(rmse_error[-1])

	plt.plot(X_axis, X_test_rmse)
	plt.show()

	# For RMSE of absolute error
	X_test_rmse = []
	X_axis = [0.025, 0.05, 0.1, 0.2, 0.5]

	train_error, test_error, rmse_error = train_abs(100, 500, 0.025)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = train_abs(100, 500, 0.05)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = train_abs(100, 500, 0.1)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = train_abs(100, 500, 0.2)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = train_abs(100, 500, 0.5)
	X_test_rmse.append(rmse_error[-1])

	plt.plot(X_axis, X_test_rmse)
	plt.show()

	# For RMSE of fourth powered error
	X_test_rmse = []
	X_axis = [0.025, 0.05, 0.1, 0.2, 0.5]

	train_error, test_error, rmse_error = train_fourth(100, 500, 0.025)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = train_fourth(100, 500, 0.05)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = train_fourth(100, 500, 0.1)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = train_fourth(100, 500, 0.2)
	X_test_rmse.append(rmse_error[-1])

	train_error, test_error, rmse_error = train_fourth(100, 500, 0.5)
	X_test_rmse.append(rmse_error[-1])

	plt.plot(X_axis, X_test_rmse)
	plt.show()






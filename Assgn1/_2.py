import _1
import numpy as np
import matplotlib.pyplot as plt
import sys


def plot(show_fig = True):
	X_train = np.load("X_train.npy")
	X_test = np.load("X_test.npy")
	Y_train = np.load("Y_train.npy")
	Y_test = np.load("Y_test.npy")

	X_train_cost = []
	X_test_cost = []
	X_axis = range(1, 10)

	line_1 = 0
	line_2 = 0
	line_3 = 0

	for DEGREE in range(1, 10):
		try:
			W = np.load("W_"+str(DEGREE)+".npy")
		except:
			print("Please run part 1 to generate weights.")
			sys.exit(1)
		cost = 0.0
		for m in range(len(X_test)):
			X = X_test[m]
			Y = Y_test[m]
			X_power = np.array([X**j for j in range(DEGREE+1)])

			Y_pred = np.dot(W, X_power)

			cost += (Y - Y_pred)**2

		cost /= 2* (m+1)
		print("test cost on degree "+str(DEGREE)+": "+str(cost))

		X_test_cost.append(cost)


		cost = 0.0
		for m in range(len(X_train)):
			X = X_train[m]
			Y = Y_train[m]
			X_power = np.array([X**j for j in range(DEGREE+1)])

			Y_pred = np.dot(W, X_power)

			cost += (Y - Y_pred)**2

		cost /= 2* (m+1)
		print("train cost on degree "+str(DEGREE)+": "+str(cost))
		X_train_cost.append(cost)

		X_here = range(100)
		X_here = np.array(X_here, dtype=np.float) / 100
		Y_here = []	
		for m in range(len(X_here)):
			X = X_here[m]
			X_power = np.array([X**j for j in range(DEGREE+1)])

			Y_pred = np.dot(W, X_power)
			Y_here.append(Y_pred)

		if show_fig:
			plt.subplot(3,3,DEGREE)
			plt.tight_layout()
			line_1 = plt.scatter(X_train, Y_train)
			line_2 = plt.scatter(X_test, Y_test, color = "green")
			line_3 = plt.plot(X_here, Y_here, color="red")
			plt.title("Degree "+str(DEGREE))

	if show_fig:
		plt.figlegend((line_1, line_2, line_3), ("Train datapoints", "Test datapoints", "Approximated function"))
		plt.show()

		plt.plot(X_axis, X_train_cost, label = "Training")
		plt.plot(X_axis, X_test_cost, color="red", label = "Testing")
		plt.xlabel("Degree")
		plt.ylabel("Loss")
		plt.legend()
		plt.show()

	return np.argmin(X_test_cost) + 1

def call_plotter(show_fig = True):
	a = input("Would you like to train (part 1) again? ")
	if (a[0] == 'y' or a[0] == 'Y'):
		a = input("Would you like to regenerate the data? ")
		if (a[0] == 'y' or a[0] == 'Y'):
			_1.generate_data()
		_1.train()
		print("\n")
	return plot(show_fig)

if __name__ == "__main__":
	call_plotter()
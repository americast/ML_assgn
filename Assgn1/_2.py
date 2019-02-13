import numpy as np
import matplotlib.pyplot as plt

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
Y_train = np.load("Y_train.npy")
Y_test = np.load("Y_test.npy")

for DEGREE in range(1, 10):
	W = np.load("W_"+str(DEGREE)+".npy")
	cost = 0.0
	for m in range(len(X_test)):
		X = X_test[m]
		Y = Y_test[m]
		X_power = np.array([X**j for j in range(DEGREE+1)])

		Y_pred = np.dot(W, X_power)

		cost += (Y - Y_pred)**2

	cost /= 2* (m+1)
	print("test cost on degree "+str(DEGREE)+": "+str(cost))

	X_here = range(100)
	X_here = np.array(X_here, dtype=np.float) / 100
	Y_here = []	
	for m in range(len(X_here)):
		X = X_here[m]
		X_power = np.array([X**j for j in range(DEGREE+1)])

		Y_pred = np.dot(W, X_power)
		Y_here.append(Y_pred)

	plt.scatter(X_train, Y_train)
	plt.plot(X_here, Y_here, color="red")
	plt.show()


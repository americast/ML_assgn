import numpy as np
import random
import matplotlib.pyplot as plt
MAX_ITER = 500
ALPHA = 0.05
NUM_DATA = 100

X = np.random.uniform(0, 1, NUM_DATA)


noise = np.random.normal(0, 0.3, NUM_DATA)

Y = (2 * np.pi * X) + noise

cutoff = int(0.8 * NUM_DATA)

X_test = X[cutoff:]
Y_test = Y[cutoff:]

X_train = X[:cutoff]
Y_train = Y[:cutoff]

np.save("X_train", X_train)
np.save("X_test", X_test)
np.save("Y_train", Y_train)
np.save("Y_test", Y_test)

for DEGREE in xrange(1, 10):
	print DEGREE



	W = np.random.uniform(0, 1, DEGREE + 1)


	for i in xrange(MAX_ITER):

		cost = 0.0
		sum_ = 0.0
		for m in xrange(cutoff):
			X = X_train[m]
			Y = Y_train[m]
			X_power = np.array([X**j for j in xrange(DEGREE+1)])

			Y_pred = np.dot(W, X_power)
			sum_ += (Y_pred - Y)
			cost += (Y - Y_pred)**2

		cost/= (2 * (m+1))
		grad = sum_ * X_power

		W -= (ALPHA/(DEGREE + 1)) * grad
		print("Cost at iter "+str(i)+"/"+str(MAX_ITER)+": "+str(cost))


	np.save("W_"+str(DEGREE), W)




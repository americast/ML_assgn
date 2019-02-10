import numpy as np
import random
import matplotlib.pyplot as plt
degree = 5
max_iter = 50
alpha = 0.05

X = np.random.uniform(0, 1, 10)


noise = np.random.normal(0, 0.3, 10)

Y = (2 * np.pi * X) + noise

# indices = range(10)
# choice_1 = random.choice(indices)
# indices.remove(choice_1)
# choice_2 = random.choice(indices)

X_test = X[8:]
Y_test = Y[8:]

X_train = X[:8]
Y_train = Y[:8]


W = np.random.uniform(0, 1, degree + 1)


for i in xrange(max_iter):

	cost = 0.0
	sum_ = 0.0
	for m in xrange(8):
		X = X_train[m]
		Y = Y_train[m]
		X_power = np.array([X**j for j in xrange(degree+1)])

		Y_pred = np.dot(W, X_power)
		sum_ += (Y_pred - Y)
		cost += (Y - Y_pred)**2

	cost/= (2 * m)
	grad = sum_ * X_power

	W -= (alpha/(degree + 1)) * grad
	print("Cost at iter "+str(i)+"/"+str(max_iter)+": "+str(cost))


import _1
import _2
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def train_abs(MAX_ITER = 5000, ALPHA = 0.05):

    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    Y_train = np.load("Y_train.npy")
    Y_test = np.load("Y_test.npy")

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


def train_fourth(MAX_ITER = 5000, ALPHA = 0.05):

    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    Y_train = np.load("Y_train.npy")
    Y_test = np.load("Y_test.npy")

    train_error = []
    test_error = []
    rmse_error = []

    for DEGREE in range(1, 10):
        print("\nDegree: "+str(DEGREE))
        W = np.random.uniform(0, 0.001, DEGREE + 1)

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
    _1.generate_data(100)
    print("\nSquared error loss\n")

    train_error, test_error, rmse_error = _1.train(500, 0.025)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))
    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = _1.train(500, 0.05)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = _1.train(500, 0.1)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = _1.train(500, 0.2)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = _1.train(500, 0.5)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    plt.plot(X_axis, X_test_rmse, label = "Squared error loss")

    print("\nAbsolute value loss\n")
    # For RMSE of absolute error
    X_test_rmse = []

    train_error, test_error, rmse_error = train_abs(500, 0.025)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = train_abs(500, 0.05)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = train_abs(500, 0.1)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = train_abs(500, 0.2)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = train_abs(500, 0.5)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    plt.plot(X_axis, X_test_rmse, label = "Absolute value loss", color = "green")

    # For RMSE of fourth powered error
    print("\nFourth power loss loss\n")
    X_test_rmse = []

    train_error, test_error, rmse_error = train_fourth(500, 0.025)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = train_fourth(500, 0.05)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = train_fourth(500, 0.1)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = train_fourth(500, 0.2)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    train_error, test_error, rmse_error = train_fourth(500, 0.5)
    print("\nProceeding with degree "+str(np.argmin(rmse_error)+1))
    X_test_rmse.append(np.min(rmse_error))

    print("\n")
    k = np.load("W_"+str(np.argmin(rmse_error)+1)+".npy")
    print("Weights: ")
    for each in k:
        print(each)
    print("\n\n")

    plt.plot(X_axis, X_test_rmse, label = "Fourth power loss", color = "red")
    plt.xlabel("Learning rate")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()






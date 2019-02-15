import _1
import matplotlib.pyplot as plt

print("\n\nData size 10: ")
train_10, test_10, rmse = _1.train(10, 500)
print("\n\nData size 100: ")
train_100, test_100, rmse = _1.train(100, 500)
print("\n\nData size 1000: ")
train_1000, test_1000, rmse = _1.train(1000, 500)
print("\n\nData size 10000: ")
train_10000, test_10000, rmse = _1.train(10000, 500)

train_erroes = [train_10[-1], train_100[-1], train_1000[-1], train_10000[-1]]
test_erroes = [test_10[-1], test_100[-1], test_1000[-1], test_10000[-1]]

X_axis = [10, 100, 1000, 10000]

plt.plot(X_axis, train_erroes, label = "Train error")
plt.plot(X_axis, test_erroes, color = "red", label = "Test error")
plt.xlabel("Total number of datapoints")
plt.ylabel("Loss")
plt.legend()
plt.show()
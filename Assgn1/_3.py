import _1
import _2
import matplotlib.pyplot as plt

n = 9
a = input("Would you like to auto compute an optimised degree? ")
if (a[0] == 'y' or a[0] == 'Y'):
	n = _2.call_plotter(show_fig = False)
else:
	a = input("Would you like to enter a degree manually? ")
	if (a[0] == 'y' or a[0] == 'Y'):
		n = input("Pl enter the degree (max 9): ")

print("Proceeding with degree "+str(n))
print("\n\nData size 10: ")
train_10, test_10, rmse = _1.train(10, 500)
print("\n\nData size 100: ")
train_100, test_100, rmse = _1.train(100, 500)
print("\n\nData size 1000: ")
train_1000, test_1000, rmse = _1.train(1000, 500)
print("\n\nData size 10000: ")
train_10000, test_10000, rmse = _1.train(10000, 500)

train_erroes = [train_10[n-1], train_100[n-1], train_1000[n-1], train_10000[n-1]]
test_erroes = [test_10[n-1], test_100[n-1], test_1000[n-1], test_10000[n-1]]

X_axis = [10, 100, 1000, 10000]

plt.plot(X_axis, train_erroes, label = "Train error")
plt.plot(X_axis, test_erroes, color = "red", label = "Test error")
plt.xlabel("Total number of datapoints")
plt.ylabel("Loss")
plt.legend()
plt.show()
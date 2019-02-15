import _1
import _2
import _4

a = int(input("Enter no. of data to be generated: "))
_1.generate_data(a)

iter_ = int(input("Enter no. iterations to perform: "))
alpha_ = float(input("Enter the learning rate to be used: "))

print("\nChoose cost function:")
print("1. Mean squared error")
print("2. Mean absolute error")
print("3. Mean fourth powered error")
a = int(input("Enter choice: "))

if (a == 1):
	_1.train(iter_, alpha_)
elif (a == 2):
	_4.train_abs(iter_, alpha_)
else:
	_4.train_fourth(iter_, alpha_)

_2.plot()
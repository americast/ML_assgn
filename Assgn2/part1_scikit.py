import pandas as pd
import numpy as np
import pudb
from sklearn import tree

def dtc(df, df_test, criterion):
	for each in df:
		col = df[each]
		unique = list(col.unique())
		mapping = {}
		for i in range(len(unique)):
			mapping[unique[i]] = i
		for i in range(len(list(col))):
			col[i] = mapping[col[i]]

		col = df_test[each]
		unique = list(col.unique())
		mapping = {}
		for i in range(len(unique)):
			mapping[unique[i]] = i
		for i in range(len(list(col))):
			col[i] = mapping[col[i]]
			
	X = []
	Y = []
	X_test = []
	Y_test = []

	for i in range(df["profitable"].count()):
		row_here = df.loc[i]
		X.append(list(row_here)[:-1])
		Y.append(list(row_here)[-1])



	clf = tree.DecisionTreeClassifier(criterion = criterion)
	clf = clf.fit(X, Y)

	acc = 0
	total = df_test["profitable"].count()
	for i in range(df_test["profitable"].count()):
		row_here = df_test.loc[i]
		X_test = list(row_here)[:-1]
		Y_test = list(row_here)[-1]

		Y_pred = clf.predict([X_test])[0]

		if Y_test == Y_pred:
			acc += 1

	# print("Acc: "+str(float(acc) / total))
	test_acc = float(acc) / total

	acc = 0
	total = df["profitable"].count()
	for i in range(df["profitable"].count()):
		row_here = df.loc[i]
		X_test = list(row_here)[:-1]
		Y_test = list(row_here)[-1]

		Y_pred = clf.predict([X_test])[0]

		if Y_test == Y_pred:
			acc += 1

	# print("Acc: "+str(float(acc) / total))

	train_acc = float(acc) / total

	return train_acc, test_acc
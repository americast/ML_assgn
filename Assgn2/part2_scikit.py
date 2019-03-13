import pandas as pd
import numpy as np
import pudb
from sklearn import tree



def create_df():
	f = open("dataset for part 2/traindata.txt", "r")
	f_label = open("dataset for part 2/trainlabel.txt", "r")
	old_docID, old_wordID = 1, 1
	old_flag = False
	df = np.zeros((1060, 3567), dtype = np.uint64)
	count = -1
	while(1):
		count += 1
		# print(count)
		# words = [0 for x in range(3566)]
		if old_flag:
			df[count, old_wordID - 1] = 1
		while(1):
			line = f.readline()
			if not line:
				break
			line = line.split()
			docID, wordID = int(line[0]), int(line[1])
			if (docID != old_docID):
				old_wordID = wordID
				old_docID = docID
				old_flag = True
				break
			df[count, wordID - 1] = 1

		if not line:
			break
		label = int(f_label.readline())

		# words.append(label)
		df[count, -1] = label
		# df = df.append([pd.Series(words)])
		# print("Appended to df")
		# pu.db


	df = pd.DataFrame(df)
	f.close()
	f_label.close()
	return df

def create_df_test():
	f = open("dataset for part 2/testdata.txt", "r")
	f_label = open("dataset for part 2/testlabel.txt", "r")
	old_docID, old_wordID = 1, 1
	old_flag = False
	df = np.zeros((707, 3567), dtype = np.uint64)
	count = -1
	while(1):
		count += 1
		# print(count)
		# words = [0 for x in range(3566)]
		if old_flag:
			df[count, old_wordID - 1] = 1
		while(1):
			line = f.readline()
			if not line:
				break
			line = line.split()
			docID, wordID = int(line[0]), int(line[1])
			if (docID != old_docID):
				old_wordID = wordID
				old_docID = docID
				old_flag = True
				break
			df[count, wordID - 1] = 1

		if not line:
			break
		label = int(f_label.readline())

		# words.append(label)
		df[count, -1] = label
		# df = df.append([pd.Series(words)])
		# print("Appended to df")
		# pu.db


	df = pd.DataFrame(df)
	f.close()
	f_label.close()
	return df

df = create_df()
df_test = create_df_test()

# for each in df:
# 	col = df[each]
# 	unique = list(col.unique())
# 	mapping = {}
# 	for i in range(len(unique)):
# 		mapping[unique[i]] = i
# 	for i in range(len(list(col))):
# 		col[i] = mapping[col[i]]

# 	col = df_test[each]
# 	unique = list(col.unique())
# 	mapping = {}
# 	for i in range(len(unique)):
# 		mapping[unique[i]] = i
# 	for i in range(len(list(col))):
# 		col[i] = mapping[col[i]]



X = []
Y = []
X_test = []
Y_test = []


children = list(df.columns)
children = [str(x) for x in children]
out = children[-1]
children = children[:-1]


for i in range(df[int(out)].count()):
	row_here = df.loc[i]
	X.append(list(row_here)[:-1])
	Y.append(list(row_here)[-1])



clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

acc = 0
total = df_test[int(out)].count()
for i in range(df_test[int(out)].count()):
	row_here = df_test.loc[i]
	X_test = list(row_here)[:-1]
	Y_test = list(row_here)[-1]

	Y_pred = clf.predict([X_test])[0]

	if Y_test == Y_pred:
		acc += 1

print("Acc: "+str(float(acc) / total))

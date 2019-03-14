import pandas as pd
import numpy as np
import json
import pudb
from copy import copy

full_dict = {}

def compute_entropy(arr):
	total = np.sum(arr)
	entropy = 0.0
	for each in arr:
		entropy -= (each / total) * np.log2(each / total)
	return entropy

def compute_ig(df, parent):
	entropy_parent = compute_entropy(list(df[int(out)].value_counts()))
	total = entropy_parent
	total_num = df[int(out)].count()
	for each in list(df[int(parent)].unique()):
		df_here = df.loc[df[int(parent)] == each]
		entropy_child = compute_entropy(list(df_here[int(out)].value_counts()))
		total -= (df_here[int(out)].count() / float(total_num)) * entropy_child
	return total

def create_dag(df, root, full_dict, depth_here, max_depth):
	# print("Root: "+(root))
	# print(df[int(root)].unique())
	# print("Depth here: "+str(depth_here))
	if len(list(df[int(root)].unique())) <= 1 or depth_here > max_depth:
		full_dict[out] = list(df[int(out)].value_counts().index)[0]
		return
	full_dict[str(root)] = {}
	for each in list(df[int(root)].unique()):
		df_here = df.loc[df[int(root)] == each].drop(int(root), axis=1)
		max_ig = -9999999
		children = list(df_here.columns)
		children.remove(int(out))
		child = children[0]
		for child_here in children:
			ig_here = compute_ig(df_here, child_here)
			if ig_here > max_ig:
				max_ig = ig_here
				child = child_here
		full_dict[str(root)][str(each)] = {}
		create_dag(df_here, child, full_dict[str(root)][str(each)], depth_here + 1, max_depth)



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

		label = int(f_label.readline())

		# words.append(label)
		df[count, -1] = label
		if not line:
			break
		# df = df.append([pd.Series(words)])
		# print("Appended to df")
		# pu.db


	df = pd.DataFrame(df)
	f.close()
	f_label.close()
	return df
	

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

		label = int(f_label.readline())

		# words.append(label)
		df[count, -1] = label
		if not line:
			break
		# df = df.append([pd.Series(words)])
		# print("Appended to df")
		# pu.db


	df = pd.DataFrame(df)
	f.close()
	f_label.close()
	return df
	
# pu.db
def infer(row, model):
	while(1):
		# print("here")
		choice = [x for x in model][0]
		# pu.db
		actual_value = row[int(choice)]
		# print("model: "+str(model))
		# print("choice: "+str(choice))
		model = model[choice]
		# pu.db
		# print()
		if choice == out:
			return model
		if (str(actual_value) not in model) or actual_value == out:
			pu.db
			return model[out]
		model = model[str(actual_value)]

print("Loading data")
df = create_df()
df_test = create_df_test()
f = open("model_part_2.json", "r")
model = json.load(f)
f.close()

children = list(df.columns)
children = [str(x) for x in children]
out = children[-1]
children = children[:-1]
child = children[0]

acc = 0
total_num = df[int(out)].count()
for i in range(df[int(out)].count()):
	row_here = df.loc[i]
	result = infer(row_here, copy(model))
	if (result == list(row_here)[-1]):
		acc+=1

print("Training accuracy: "+str(float(acc)/total_num))


children = list(df_test.columns)
children = [str(x) for x in children]
out = children[-1]
children = children[:-1]
child = children[0]

acc = 0
total_num = df_test[int(out)].count()
for i in range(df_test[int(out)].count()):
	row_here = df_test.loc[i]
	result = infer(row_here, copy(model))
	if (result == list(row_here)[-1]):
		acc+=1

print("Test accuracy: "+str(float(acc)/total_num))
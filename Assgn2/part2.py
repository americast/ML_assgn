import pandas as pd
import numpy as np
import json
import pudb

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
	print("Depth here: "+str(depth_here))
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
	
print("Loading data")
df = create_df()
# pu.db

print("Finding out root")
max_ig = -9999999
children = list(df.columns)
children = [str(x) for x in children]
out = children[-1]
children = children[:-1]
child = children[0]
for child_here in children:
	print(child_here)
	ig_here = compute_ig(df, child_here)
	if ig_here > max_ig:
		max_ig = ig_here
		child = child_here

print("Creating dag")
create_dag(df, child, full_dict, depth_here = 1, max_depth = 2)
# pu.db
print(json.dumps(full_dict, sort_keys=True, indent=4))

f = open("model_part_2.json", "w")
json.dump(full_dict, f, sort_keys = True, indent = 4)
f.close()
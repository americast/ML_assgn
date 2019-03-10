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
	entropy_parent = compute_entropy(list(df["profitable"].value_counts()))
	total = entropy_parent
	total_num = df["profitable"].count()
	for each in list(df[parent].unique()):
		df_here = df.loc[df[parent] == each]
		entropy_child = compute_entropy(list(df_here["profitable"].value_counts()))
		total -= (df_here["profitable"].count() / float(total_num)) * entropy_child
	return total

def create_dag(df, root, full_dict):
	# print("Root: "+str(root))
	# print(df[root].unique())
	if len(list(df[root].unique())) <= 1:
		full_dict["profitable"] = list(df["profitable"].value_counts().index)[0]
		return
	full_dict[str(root)] = {}
	for each in list(df[root].unique()):
		df_here = df.loc[df[root] == each].drop(root, axis=1)
		max_ig = -9999999
		children = list(df_here.columns)
		children.remove("profitable")
		child = children[0]
		for child_here in children:
			ig_here = compute_ig(df_here, child_here)
			if ig_here > max_ig:
				max_ig = ig_here
				child = child_here
		full_dict[str(root)][str(each)] = {}
		create_dag(df_here, child, full_dict[str(root)][str(each)])



	

df = pd.read_csv("dataset for part 1 - Training Data.csv")

max_ig = -9999999
children = list(df.columns)
children.remove("profitable")
child = children[0]
for child_here in children:
	ig_here = compute_ig(df, child_here)
	if ig_here > max_ig:
		max_ig = ig_here
		child = child_here

create_dag(df, child, full_dict)

print(json.dumps(full_dict, sort_keys=True, indent=4))

f = open("model_part_1a.json", "w")
json.dump(full_dict, f, sort_keys = True, indent = 4)
f.close()
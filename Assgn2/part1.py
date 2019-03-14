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
	entropy_parent = compute_entropy(list(df[out].value_counts()))
	total = entropy_parent
	total_num = df[out].count()
	for each in list(df[parent].unique()):
		df_here = df.loc[df[parent] == each]
		entropy_child = compute_entropy(list(df_here[out].value_counts()))
		total -= (df_here[out].count() / float(total_num)) * entropy_child
	return total

def gini(arr):
	total = np.sum(arr)
	gini = 1.0
	for each in arr:
		gini -= (float(each) / total) ** 2
	return gini

def compute_gini(df, parent):
	gini_split = 0.0
	total_num = df[out].count()
	for each in list(df[parent].unique()):
		df_here = df.loc[df[parent] == each]
		gini_here = gini(list(df_here[out].value_counts()))
		gini_split += (df_here[out].count() / float(total_num)) * gini_here
	return gini_split



def create_dag(df, root, full_dict, imp_func = compute_ig):
	# print("Root: "+str(root))
	# print(df[root].unique())
	if len(list(df[root].unique())) <= 1 or len(list(df[out].unique())) <= 1:
		full_dict[out] = list(df[out].value_counts().index)[0]
		return
	full_dict[str(root)] = {}
	for each in list(df[root].unique()):
		df_here = df.loc[df[root] == each].drop(root, axis=1)
		max_ig = -9999999
		# pu.db
		children = list(df_here.columns)
		children.remove(out)
		child = children[0]
		for child_here in children:
			if imp_func == compute_ig:
				ig_here = imp_func(df_here, child_here)
				if ig_here > max_ig:
					max_ig = ig_here
					child = child_here
			else:
				ig_here = imp_func(df_here, child_here)
				if ig_here < max_ig:
					max_ig = ig_here
					child = child_here

		full_dict[str(root)][str(each)] = {}
		print(root)
		print(each)
		print()
		create_dag(df_here, child, full_dict[str(root)][str(each)])



	

df = pd.read_csv("dataset for part 1 - Training Data.csv")

max_ig = -9999999
children = list(df.columns)
out = children[-1]
children = children[:-1]
child = children[0]
for child_here in children:
	ig_here = compute_ig(df, child_here)
	if ig_here > max_ig:
		max_ig = ig_here
		child = child_here

create_dag(df, child, full_dict, compute_gini)

print(json.dumps(full_dict, sort_keys=True, indent=4))

f = open("model_part_1a.json", "w")
json.dump(full_dict, f, sort_keys = True, indent = 4)
f.close()
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
	# pu.db
	entropy_parent = compute_entropy(np.unique(df[..., -1], return_counts=True)[1])
	total = entropy_parent
	total_num = df.shape[0]
	try:
		for each in np.unique(df[...,int(parent)]):
			df_here = df[df[...,int(parent)] == each]
			entropy_child = compute_entropy(np.unique(df_here[..., -1], return_counts=True)[1])
			total -= (df_here.shape[0] / float(total_num)) * entropy_child
	except:
		pu.db
	return total

def create_dag(df, root, full_dict, depth_here, max_depth, out):
	# print("Root: "+(root))
	# print(df[int(root)].unique())
	# if depth_here == 2:
	print("Depth here: "+str(depth_here))
	if df.shape[0] == 0 or depth_here > max_depth:
		try:
			full_dict[out] = list(df[int(out)].value_counts().index)[0]
		except:
			full_dict[out] = 2
		print("returned from here")
		return
	full_dict[str(root)] = {}
	for each in [0, 1]:
		df_here = df.loc[df[int(root)] == each].drop(int(root), axis=1)
		max_ig = float('-inf')
		children_org = list(df_here.columns)
		children = range(len(list(df_here.columns)))
		children = children[:-1]
		child = children[0]
		df_np = np.array(df_here)
		for i in range(len(children)):
			child_here = children[i]
			ig_here = compute_ig(df_np, child_here)
			if ig_here > max_ig:
				max_ig = ig_here
				child = children_org[i]
		full_dict[str(root)][str(each)] = {}
		# pu.db
		create_dag(df_here, child, full_dict[str(root)][str(each)], depth_here + 1, max_depth, out)

def prun_dag(df, full_dict, depth, max_depth, out, models):
	choice = [x for x in full_dict][0]
	df_here = df[int(choice)]

if __name__=="__main__":

	print("Loading data")
	df = create_df()
	# pu.db

	print("Finding out root")
	max_ig = float('-inf')
	children = list(df.columns)
	children = [str(x) for x in children]
	out = children[-1]
	children = children[:-1]
	child = children[0]
	df_np = np.array(df)
	for child_here in children:
		print(child_here)
		ig_here = compute_ig(df_np, child_here)
		if ig_here > max_ig:
			max_ig = ig_here
			child = child_here

	print("Creating dag")
	create_dag(df, child, full_dict, depth_here = 1, max_depth = 10)
	# pu.db
	print(json.dumps(full_dict, sort_keys=True, indent=4))

	f = open("model_part_2.json", "w")
	json.dump(full_dict, f, sort_keys = True, indent = 4)
	f.close()

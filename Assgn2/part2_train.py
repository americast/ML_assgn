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

def compute_ig(df_col_parent, df_col_out):
	# pu.db
	entropy_parent = compute_entropy(np.unique(df_col_out, return_counts=True)[1])
	total = entropy_parent
	total_num = len(df_col_parent)
	for each in np.unique(df_col_parent):
		df_here_col_out = df_col_out[df_col_parent == each]
		entropy_child = compute_entropy(np.unique(df_here_col_out, return_counts=True)[1])
		total -= (len(df_here_col_out) / float(total_num)) * entropy_child
	return total

def create_dag(df, root, full_dict, depth_here, max_depth, out, considered_children = []):
	# print("Root: "+(root))
	# print(df[int(root)].unique())
	# if depth_here == 2:
	print("Depth here: "+str(depth_here))
	if len(np.unique(df[..., int(root)])) <=1 or len(np.unique(df[..., -1])) <=1 or df.shape[0] == 0 or depth_here > max_depth:
		# pu.db
		try:
			full_dict[str(out)] = int(np.bincount(df[..., -1].astype(np.intp)).argmax())
		except:
			full_dict[str(out)] = 2
		print("returned from here")
		return
	full_dict[str(root)] = {}
	updated_cc = considered_children + [int(root)]
	for each in [0, 1]:
		df_here = df[df[..., int(root)] == each]
		max_ig = float('-inf')
		# children_org = list(df_here.columns)
		# children = range(len(list(df_here.columns)))
		# children = children[:-1]
		child = 0
		# df_np = np.array(df_here)
		for i in range(df.shape[1] - 1):
			# child_here = children[i]
			if i in updated_cc: continue
			ig_here = compute_ig(df_here[..., i], df_here[..., -1])
			if ig_here > max_ig:
				max_ig = ig_here
				child = i
		full_dict[str(root)][str(each)] = {}
		full_dict[str(root)][str(each)]["__result__"] = int(np.bincount(df_here[..., -1].astype(np.intp)).argmax())
		# pu.db
		create_dag(df_here, child, full_dict[str(root)][str(each)], depth_here + 1, max_depth, out, updated_cc)



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

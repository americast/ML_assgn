import pandas as pd
import numpy as np
import pudb

df = pd.read_csv("AAAI.csv")

import part1_mod
import part2
import part3
import warnings
warnings.filterwarnings("ignore")


def compute_entropy(arr):
	total = np.sum(arr)
	entropy = 0.0
	for each in arr:
		entropy -= (each / total) * np.log2(each / total)
	return entropy

h_y = compute_entropy(list(df["High-Level Keyword(s)"].value_counts()))

def entropy_clus(clus):
	arr = []
	for each in clus:
		arr.append(len(each))
	return compute_entropy(arr)

def entropy_class(arr):
	df_here = df.loc[list(np.array(arr) - 1)]
	# pu.db
	return compute_entropy(list(df_here["High-Level Keyword(s)"].value_counts()))
	# pu.db

def NMF(clus):
	# h_y = entropy_class(clus)
	h_c = entropy_clus(clus)

	I_y_c = h_y


	total = 0

	for each in clus:
		total += len(each)

	for each in clus:
		I_y_c -= (float(len(each)) / total) * entropy_class(each)
	# pu.db
	return 2 * float(I_y_c) / (h_y + h_c)

clus_1 = part1_mod.hier_clus(df)
clus_2 = part1_mod.hier_clus(df, "single")
clus_3 = part2.graph_clus(df, 0.1)
# clus_4 = part3.high_level_clus(df)

# NMF_4 = NMF(clus_4)
NMF_1 = NMF(clus_1)
NMF_2 = NMF(clus_2)
NMF_3 = NMF(clus_3)

print("NMF of:\nHierarchical clustering (complete): "+str(NMF_1)+"\nHierarchical clustering (single): "+str(NMF_2))
print("Graph clustering: "+str(NMF_3))

# pu.db
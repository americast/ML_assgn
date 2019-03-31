import pandas as pd
import numpy as np
import pudb

df = pd.read_csv("AAAI.csv")

import part1
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

def NMI(clus):
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

print("Performing complete hierarchical clustering: ")
clus_1 = part1_mod.hier_clus(df)
NMI_1 = NMI(clus_1)
print("NMI: "+str(NMI_1)+"\n")

print("Performing single hierarchical clustering: ")
clus_2 = part1_mod.hier_clus(df, "single")
NMI_2 = NMI(clus_2)
print("NMI: "+str(NMI_2)+"\n")

print("Performing graph clustering (with threshold at 0.1): ")
clus_3 = part2.graph_clus(df, 0.1)
NMI_3 = NMI(clus_3)
print("NMI: "+str(NMI_3)+"\n")
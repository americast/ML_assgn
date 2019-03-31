###########
# ML Assgn 3
# Sayan Sinha
# 16CS10048
###########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import pudb

def high_level_clus(df):
	topics = []
	node_lens = []
	all_dict = {}
	counter = 0
	# pu.db
	# G = nx.Graph()
	for each in df["High-Level Keyword(s)"]:
		counter += 1
		if each not in all_dict:
			all_dict[each] = []
		all_dict[each].append(counter)
		# node_lens.append(len(topics_here))
		# G.add_node(counter)

	all_nodes = []
	for each in all_dict:
		all_nodes.append(all_dict[each])

	return all_nodes
	# pu.db


if __name__ == "__main__":
	df = pd.read_csv("AAAI.csv")
	print(high_level_clus(df))

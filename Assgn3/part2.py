import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pudb

DIST_THRESH = 0

def jaccard_coef(a, b):
	inter = len(set(a).intersection(b))
	union = len(set(a).union(set(b)))
	return float(inter) / union

def distance(clust_choice_a, clust_choice_b, linkage="complete"):
	max_ = float('-inf')
	min_ = float('inf')
	for i in range(node_lens[clust_choice_a]):
		for j in range(node_lens[clust_choice_b]):
			try:
				coef_here = jaccard_coef(all_dict[str(node_list[clust_choice_a][i])], all_dict[str(node_list[clust_choice_b][j])])
			except: pu.db
			if (coef_here > max_):
				max_ = coef_here
			if (coef_here < min_):
				min_ = coef_here

	if linkage == "complete":
		return max_
	else:
		return min_

df = pd.read_csv("AAAI.csv")

topics = []
node_lens = []
all_dict = {}
counter = 0
G = nx.Graph()
for each in df["Topics"]:
	counter += 1
	topics_here = each.split("\n")
	topics.extend(topics_here)

	all_dict[str(counter)] = topics_here
	node_lens.append(len(topics_here))
	G.add_node(counter)


topics = set(topics)

node_list = range(1, counter + 1)

# for i in range(len(node_list)):
# 	for j in range(len(node_list)):
# 		node_list[i].append(-1)

tot_count = len(node_list)

for i in range(tot_count):
	for j in range(i + 1, tot_count):
		dist_here = jaccard_coef(all_dict[str(i + 1)], all_dict[str(j+1)])
		# print(dist_here)
		if (dist_here > DIST_THRESH):
			G.add_edge(i + 1, j + 1)

def second_elem(a):
	return a[1]
# pu.db
iter_ = 1
print()
while(1):
	clusters = list(nx.connected_component_subgraphs(G))
	# pu.db
	num_clusters = len(clusters)

	print("iter_no: "+str(iter_)+", size: "+str(num_clusters), end = "\r")
	iter_ += 1
	if num_clusters >= 9:
		break

	centralities = list(nx.edge_betweenness_centrality(G).items())
	# pu.db
	centralities.sort(key = second_elem, reverse = True)
	req_edges = centralities[0][0]

	# print(req_edges)
	G.remove_edge(req_edges[0], req_edges[1])
print()
nx.draw(G)
plt.show()
# pu.db


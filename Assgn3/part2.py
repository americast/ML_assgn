###########
# ML Assgn 3
# Sayan Sinha
# 16CS10048
###########

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# import pudb
import part1
import warnings
warnings.filterwarnings("ignore")

DIST_THRESH = 0

def graph_clus(df, DIST_THRESH = DIST_THRESH):
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
			dist_here = part1.jaccard_coef(all_dict[str(i + 1)], all_dict[str(j+1)])
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

		print("iter_no: "+str(iter_)+", clusters: "+str(num_clusters), end = "\r")
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
	plt.show("Clusters map")

	clusters = list(nx.connected_component_subgraphs(G))
	all_nodes = []
	for each in clusters:  
	    all_nodes.append(list(each.nodes))  
	# pu.db
	return all_nodes

	# pu.db

if __name__ == "__main__":
	df = pd.read_csv("AAAI.csv")

	try:
		thresh = float(input("Enter threshold: "))
	except:
		thresh = DIST_THRESH

	clus = graph_clus(df, thresh)
	titles = list(df["Title"])
	clus_names = []
	for each in clus:       
	    clus_names.append([])                                        
	    for every in each:
	        clus_names[-1].append(titles[every - 1])
	            
	count = 1

	for each in clus_names:   
		print("Cluster no "+str(count)+":" )                         
		print(each)       
		print()                                     
		count+=1


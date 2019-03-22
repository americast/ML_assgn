import pandas as pd
import numpy as np
import pudb

def jaccard_coef(a, b):
	inter = len(set(a).intersection(b))
	union = len(set(a).union(set(b)))
	return float(inter) / union

def distance(clust_choice_a, clust_choice_b, linkage="complete"):
	max_ = float('-inf')
	min_ = float('inf')
	for i in range(node_lens[clust_choice_a]):
		for j in range(node_lens[clust_choice_b]):
			coef_here = jaccard_coef(all_dict[str(node_list[clust_choice_a][i])], all_dict[str(node_list[clust_choice_b][j])])
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
all_dict = {}
counter = 0
for each in df["Topics"]:
	counter += 1
	topics_here = each.split("\n")
	topics.extend(topics_here)

	all_dict[str(counter)] = topics_here

topics = set(topics)


node_list = [[x] for x in range(1, counter + 1)]

for i in range(len(node_list)):
	for j in range(len(node_list)):
		node_list[i].append(-1)

node_lens = [1 for x in range(len(node_list))]

prox_mat = np.zeros((len(node_list), len(node_list)))

tot_count = len(node_list)

for i in range(tot_count):
	for j in range(i + 1, tot_count):
		prox_mat[i, j] = distance(i, j)
		prox_mat[j, i] = prox_mat[i, j]

iter_ = 1
print()
while(1):
	print("iter_no: "+str(iter_)+", size: "+str(tot_count))
	iter_ += 1
	if tot_count <= 9:
		break
	tot_count-=1

	min_dist = float('inf')

	choice_1 = 0
	choice_2 = 0

	for i in range(len(node_list)):
		for j in range(i+1, len(node_list)):
			if node_list[i] == node_list[j]:
				continue
			try:
				dist_here = prox_mat[node_list[i][0] - 1, node_list[j][0] - 1]
			except:
				pu.db
			if min_dist > dist_here:
				min_dist = dist_here
				choice_1 = i
				choice_2 = j

	for each_item in node_list[choice_2]:
		if each_item == -1:
			break
		try:	
			node_list[choice_1][node_lens[choice_1]] = each_item
			node_lens[choice_1] += 1
		except:
			pu.db

	node_list[choice_2] = node_list[choice_1]

	for i in range(len(node_list)):
		if i not in node_list[choice_1]:
			prox_mat[node_list[choice_1][0] - 1, i] = distance(choice_1, choice_2)
			prox_mat[i, node_list[choice_1][0] - 1] = prox_mat[node_list[choice_1][0] - 1, i]

		for j in node_list[choice_1]:
			if j == -1:
				break
			prox_mat[j - 1, i] = prox_mat[node_list[choice_1][0] - 1, i]
			prox_mat[i, j - 1] = prox_mat[i, node_list[choice_1][0] - 1]



print()

pu.db

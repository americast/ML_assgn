import pandas as pd
import numpy as np
import pudb

def jaccard_coef(a, b):
	inter = len(set(a).intersection(b))
	union = len(set(a).union(set(b)))
	# print(a)
	# print(b)
	return float(inter) / union

# def distance(clust_choice_a, clust_choice_b, all_dict, node_list, node_lens, linkage="complete"):
# 	max_ = float('-inf')
# 	min_ = float('inf')
# 	for i in range(node_lens[clust_choice_a]):
# 		for j in range(node_lens[clust_choice_b]):
# 			coef_here = jaccard_coef(all_dict[str(node_list[clust_choice_a][i])], all_dict[str(node_list[clust_choice_b][j])])
# 			print("coef here: ", coef_here)
# 			if (coef_here > max_):
# 				max_ = coef_here
# 			if (coef_here < min_):
# 				min_ = coef_here

# 	if linkage == "complete":
# 		return max_
# 	else:
# 		return min_

def unique(list1): 
	# intilize a null list 
	unique_list = [] 
	  
	# traverse for all elements 
	for x in list1: 
	    # check if exists in unique_list or not 
	    if x not in unique_list: 
	        unique_list.append(x) 
	return unique_list


def hier_clus(df, linkage="complete"):
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
			prox_mat[i, j] = jaccard_coef(all_dict[str(node_list[i][0])], all_dict[str(node_list[j][0])])
			prox_mat[j, i] = prox_mat[i, j]

	iter_ = 1
	print()
	while(1):
		print("iter_no: "+str(iter_)+", size: "+str(tot_count)+"    ", end="\r")
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
					# print("Here")
					continue
				try:
					dist_here = prox_mat[i, j]
				except:
					pu.db
				if min_dist > dist_here:
					min_dist = dist_here
					choice_1 = i
					choice_2 = j

		choice_1_s = []
		choice_2_s = []

		for k in range(len(node_list)):
			if node_list[k] == node_list[choice_1]:
				choice_1_s.append(k)
			if node_list[k] == node_list[choice_2]:
				choice_2_s.append(k)
		# print("choice_1_s: ", choice_1_s)
		# print("choice_2_s: ", choice_2_s)

		for i in range(len(node_list)):
			if i not in choice_1_s and i not in choice_2_s:
				if linkage == "complete":
					prox_mat[choice_1, i] = max(prox_mat[choice_1, i], prox_mat[choice_2, i])
				else:
					prox_mat[choice_1, i] = min(prox_mat[choice_1, i], prox_mat[choice_2, i])

				for l in choice_2_s:
					prox_mat[l, i] = prox_mat[choice_1, i]
					prox_mat[i, l] = prox_mat[choice_1, i]

				for l in choice_1_s:
					prox_mat[i, l] = prox_mat[choice_1, i]


		# if (iter_ == 138):
		# 	pu.db

		for each_item in node_list[choice_2]:
			if each_item == -1:
				break
			try:
				node_list[choice_1][node_lens[choice_1]] = each_item
				node_lens[choice_1] += 1
			except:
				pu.db

		for l in choice_1_s:
			node_list[l] = node_list[choice_1]
			node_lens[l] = node_lens[choice_1]


		for l in choice_2_s:
			node_list[l] = node_list[choice_1]
			node_lens[l] = node_lens[choice_1]


	print()

	node_list = unique(node_list)

	for i in range(len(node_list)):
		node_list[i] = node_list[i][:np.argmin(node_list[i])]

	return node_list

if __name__=="__main__":
	df = pd.read_csv("AAAI.csv")
	print("Choose linkage type:\n1) Single\n2) Complete\n")
	choice = int(input("Enter choice: "))
	if choice == 1:
		print(hier_clus(df, "single"))
	else:
		print(hier_clus(df))
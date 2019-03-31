import pandas as pd
import numpy as np
# import pudb

def jaccard_coef(a, b):
	inter = len(set(a).intersection(b))
	union = len(set(a).union(set(b)))
	return float(inter) / union

def proximity(clust_a, clust_b, node_lens_i, node_lens_j, all_dict, linkage="complete"):
	max_ = float('-inf')
	min_ = float('inf')
	for i in range(node_lens_i):
		for j in range(node_lens_j):
			coef_here = jaccard_coef(all_dict[str(clust_a[i])], all_dict[str(clust_b[j])])
			if (coef_here > max_):
				max_ = coef_here
			if (coef_here < min_):
				min_ = coef_here

	if linkage == "complete":
		return min_
	else:
		return max_

def unique(list1): 
	# intilize a null list 
	unique_list = [] 
	  
	# traverse for all elements 
	for x in list1: 
	    # check if exists in unique_list or not 
	    if x not in unique_list: 
	        unique_list.append(x) 
	return unique_list

def hier_clus(df, linkage = "complete"):
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
	node_lens = [1 for x in range(counter)]

	for i in range(len(node_list)):
		for j in range(len(node_list) - 1):
			node_list[i].append(-1)

	iter_ = 1
	print()
	while(1):
		print("iter_no: "+str(iter_)+", clusters: "+str(len(node_list))+"   ", end="\r")
		iter_ += 1
		if len(node_list) <= 9:
			break
		min_dist = float('-inf')

		choice_1 = 0
		choice_2 = 0

		for i in range(len(node_list)):
			for j in range(i+1, len(node_list)):
				dist_here = proximity(node_list[i], node_list[j], node_lens[i], node_lens[j], all_dict, linkage)
				if min_dist < dist_here:
					min_dist = dist_here
					choice_1 = i
					choice_2 = j


		for each_item in node_list[choice_1]:
			if each_item == -1:
				break
			node_list[choice_2][node_lens[choice_2]] = each_item
			node_lens[choice_2] += 1

		# node_list[choice_1].extend(node_list[choice_2])
		if linkage != "complete":
			node_list = node_list[:choice_1] + node_list[choice_1 + 1:]
			node_lens = node_lens[:choice_1] + node_lens[choice_1 + 1:]
		else:
			node_list.append(node_list[choice_2])
			node_lens.append(node_lens[choice_2])
			node_list = node_list[:choice_2] + node_list[choice_2+1:]
			node_lens = node_lens[:choice_2] + node_lens[choice_2+1:]
			node_list = node_list[:choice_1] + node_list[choice_1 + 1:]
			node_lens = node_lens[:choice_1] + node_lens[choice_1 + 1:]


	print("\n")

	node_list = unique(node_list)

	for i in range(len(node_list)):
		node_list[i] = node_list[i][:np.argmin(node_list[i])]

	# print("Final length: "+str(len(node_list)))

	return node_list

	# pu.db

if __name__ == "__main__":
	df = pd.read_csv("AAAI.csv")
	print("Choose linkage type:\n1) Single\n2) Complete\n")
	choice = int(input("Enter choice: "))
	clus = []
	if choice == 1:
		clus = hier_clus(df, "single")
	else:
		clus = hier_clus(df)

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


	# pu.db
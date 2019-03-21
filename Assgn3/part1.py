import pandas as pd
import pudb

def jaccard_coef(a, b):
	inter = len(set(a).intersection(b))
	union = len(set(a).union(set(b)))
	return float(inter) / union

def distance(clust_a, clust_b, linkage="complete"):
	max_ = float('-inf')
	min_ = float('inf')
	for i in range(len(clust_a)):
		for j in range(len(clust_b)):
			coef_here = jaccard_coef(all_dict[str(clust_a[i])], all_dict[str(clust_b[j])])
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

iter_ = 1
print()
while(1):
	print("iter_no: "+str(iter_)+", size: "+str(len(node_list)), end="\r")
	iter_ += 1
	if len(node_list) <= 9:
		break
	min_dist = float('inf')

	choice_1 = 0
	choice_2 = 0

	for i in range(len(node_list)):
		for j in range(i+1, len(node_list)):
			dist_here = distance(node_list[i], node_list[j])
			if min_dist < dist_here:
				min_dist = dist_here
				choice_1 = i
				choice_2 = j

	node_list[choice_1].extend(node_list[choice_2])
	node_list = node_list[:j] + node_list[j+1:]

print()

pu.db

import pandas as pd
import pudb

def jaccard_coef(a, b):
	inter = len(set(a).intersection(b))
	union = len(set(a).union(set(b)))
	return float(inter) / union

def distance(clust_a, clust_b, linkage="complete"):
	i = 0
	j = 0
	max_ = float('-inf')
	min_ = float('inf')
	for i in range(len(clust_a)):
		for j in range(len(clust_b)):
			coef_here = jaccard_coef(clust_a[i], clust_b[i])
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

pu.db

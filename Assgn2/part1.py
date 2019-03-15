import part1_train
import part1_test
import part1_scikit
import pandas as pd
import numpy as np
import json
from copy import copy

full_dict = {}

df = pd.read_csv("dataset for part 1 - Training Data.csv")
df_test = pd.read_csv("dataset for part 1 - Test Data.csv")

max_ig = float('-inf')
children = list(df.columns)
out = children[-1]
children = children[:-1]
child = children[0]
for child_here in children:
	ig_here = part1_train.compute_ig(df, child_here, out)
	if ig_here > max_ig:
		max_ig = ig_here
		child = child_here

part1_train.create_dag(df, child, full_dict, "compute_ig", out)
# model = json.dumps(full_dict, sort_keys=True, indent=4)
model = full_dict
f = open("model_part_1a_ig.json", "w")
json.dump(full_dict, f, sort_keys = True, indent = 4)
f.close()


df = pd.read_csv("dataset for part 1 - Training Data.csv")

acc = 0
total_num = df["profitable"].count()



for i in range(df["profitable"].count()):
	row_here = df.loc[i]
	result = part1_test.infer(row_here, copy(model))
	# print(result)
	if (result == list(row_here)[-1]):
		acc+=1

print("Model training accuracy using information gain: "+str(float(acc)/total_num))


acc = 0
total_num = df_test["profitable"].count()


for i in range(df_test["profitable"].count()):
	row_here = df_test.loc[i]
	result = part1_test.infer(row_here, copy(model))
	# print(result)
	if (result == list(row_here)[-1]):
		acc+=1

print("Model test accuracy using information gain: "+str(float(acc)/total_num))

train_acc_scikit, test_acc_scikit = part1_scikit.dtc(df, df_test, "entropy")

print("Scikit training accuracy using information gain: "+str(train_acc_scikit))
print("Scikit test accuracy using information gain: "+str(test_acc_scikit))
print()



df = pd.read_csv("dataset for part 1 - Training Data.csv")
df_test = pd.read_csv("dataset for part 1 - Test Data.csv")

max_ig = float('-inf')
children = list(df.columns)
out = children[-1]
children = children[:-1]
child = children[0]
for child_here in children:
	ig_here = part1_train.compute_gini(df, child_here, out)
	if ig_here < max_ig:
		max_ig = ig_here
		child = child_here

part1_train.create_dag(df, child, full_dict, "compute_gini", out)
# model = json.dumps(full_dict, sort_keys=True, indent=4)
model = full_dict
f = open("model_part_1a_gini.json", "w")
json.dump(full_dict, f, sort_keys = True, indent = 4)
f.close()


df = pd.read_csv("dataset for part 1 - Training Data.csv")

acc = 0
total_num = df["profitable"].count()



for i in range(df["profitable"].count()):
	row_here = df.loc[i]
	result = part1_test.infer(row_here, copy(model))
	# print(result)
	if (result == list(row_here)[-1]):
		acc+=1

print("Model training accuracy using gini split: "+str(float(acc)/total_num))


acc = 0
total_num = df_test["profitable"].count()


for i in range(df_test["profitable"].count()):
	row_here = df_test.loc[i]
	result = part1_test.infer(row_here, copy(model))
	# print(result)
	if (result == list(row_here)[-1]):
		acc+=1

print("Model test accuracy using gini split: "+str(float(acc)/total_num))

train_acc_scikit, test_acc_scikit = part1_scikit.dtc(df, df_test, "gini")

print("Scikit training accuracy using gini split: "+str(train_acc_scikit))
print("Scikit test accuracy using gini split: "+str(test_acc_scikit))
print()


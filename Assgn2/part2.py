import numpy as np
import part2_loader
import part2_train
import part2_test
from copy import copy
import pudb

MAX_DEPTH = 4

print("Loading training data")
df = part2_loader.create_df()


print("Finding out root")
max_ig = float('-inf')
children = list(df.columns)
children = [str(x) for x in children]
out = children[-1]
children = children[:-1]
child = children[0]
df_np = np.array(df)
print()
for child_here in children:
	print(str(int(child_here) + 1) +" / "+str(len(children)), end = "\r")
	ig_here = part2_train.compute_ig(df_np, child_here)
	if ig_here > max_ig:
		max_ig = ig_here
		child = child_here
print()
full_dict = {}
print("Learning the DT")
models = [x for x in range(MAX_DEPTH)]
part2_train.create_dag(df, child, full_dict, 1, MAX_DEPTH, out, models)

print("Loading testing data")
df_test = part2_loader.create_df("test")

train_acc = []
test_acc = []
depth = 0

# pu.db
for model in models:
	depth += 1
	print("Depth: "+str(depth))
	children = list(df.columns)
	children = [str(x) for x in children]
	out = children[-1]
	children = children[:-1]
	child = children[0]

	acc = 0
	total_num = df[int(out)].count()
	for i in range(df[int(out)].count()):
		row_here = df.loc[i]
		result = part2_test.infer(row_here, copy(model), out)
		if (result == list(row_here)[-1]):
			acc+=1

	print("Train accuracy: "+str(float(acc)/total_num))
	train_acc.append(float(acc)/total_num)

	children = list(df_test.columns)
	children = [str(x) for x in children]
	out = children[-1]
	children = children[:-1]
	child = children[0]

	acc = 0
	total_num = df_test[int(out)].count()
	for i in range(df_test[int(out)].count()):
		row_here = df_test.loc[i]
		result = part2_test.infer(row_here, copy(model), out)
		if (result == list(row_here)[-1]):
			acc+=1

	test_acc.append(float(acc)/total_num)
	print("Test accuracy: "+str(float(acc)/total_num))
	print()

pu.db
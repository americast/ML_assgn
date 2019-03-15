import numpy as np
import part2_loader
import part2_train
import part2_test
import part2_scikit
from copy import copy
import pudb
import json
import pretty_print

MAX_DEPTH = 3

print("What would you like to do?")
print("1. Generate trees upto a max depth")
print("2. Generate a single tree with a max depth")
choice = int(input())

MAX_DEPTH = int(input("\nEnter max depth: "))

if choice == 1:

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
	threads_here = []
	for child_here in children:
		print(str(int(child_here) + 1) +" / "+str(len(children)), end = "\r")
		# threads_here.append(threading.Thread(target=part2_train.compute_ig, args=(df_np, child_here)))
		ig_here = part2_train.compute_ig(df_np[...,int(child_here)], df_np[..., -1])
		# threads_here[-1].start()
		if ig_here > max_ig:
			max_ig = ig_here
			child = child_here
	# for each in threads_here:
	# 	each.join()
	print("Max IG was: ", max_ig)

	print("Learning the DT")

	models = [0 for x in range(MAX_DEPTH)]
	for max_depth in range(1, MAX_DEPTH + 1):
		full_dict = {}
		np_df = np.array(df)
		part2_train.create_dag(np_df, child, full_dict, 1, max_depth, out)

		models[max_depth - 1] = full_dict
		# part2_train.prun_dags(df, full_dict, 1, MAX_DEPTH, out, models)

	print("Loading testing data")
	df_test = part2_loader.create_df("test")

	train_acc = []
	test_acc = []
	depth = 0

	f_label = open("dataset for part 2/words.txt", "r")
	mapping = []
	while(True):
		line = f_label.readline()
		if not line:
			break
		mapping.append(line)

	f_label.close()
	# pu.db
	for model in models:
		depth += 1
		# pu.db
		f = open("model_part_2_depth_"+str(depth)+".json", "w")
		json.dump(model, f)
		f.close()
		print("############################")
		print("Depth: "+str(depth))
		print("############################")
		pretty_print.pretty_print_part(model, "3566", mapping = mapping)
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

	train_acc_scikit, test_acc_scikit = part2_scikit.dtc(df, df_test, "entropy", MAX_DEPTH)


	import matplotlib.pyplot as plt

	plt.plot(train_acc, label = "Model train acc")
	plt.plot(test_acc, label = "Model test acc")
	plt.legend()
	plt.xlabel("Depth")
	plt.ylabel("Accuracy")
	plt.title("Model train and test accuracies")
	plt.show()


	plt.plot(train_acc_scikit, label = "Scikit train acc")
	plt.plot(test_acc_scikit, label = "Scikit test acc")
	plt.legend()
	plt.xlabel("Depth")
	plt.ylabel("Accuracy")
	plt.title("Scikit train and test accuracies")
	plt.show()

	# pu.db
else:


	print("Loading training data")
	df = part2_loader.create_df()

	print("Loading testing data")
	df_test = part2_loader.create_df("test")



	print("Finding out root")
	max_ig = float('-inf')
	children = list(df.columns)
	children = [str(x) for x in children]
	out = children[-1]
	children = children[:-1]
	child = children[0]
	df_np = np.array(df)
	print()
	threads_here = []
	for child_here in children:
		print(str(int(child_here) + 1) +" / "+str(len(children)), end = "\r")
		# threads_here.append(threading.Thread(target=part2_train.compute_ig, args=(df_np, child_here)))
		ig_here = part2_train.compute_ig(df_np[...,int(child_here)], df_np[..., -1])
		# threads_here[-1].start()
		if ig_here > max_ig:
			max_ig = ig_here
			child = child_here
	# for each in threads_here:
	# 	each.join()
	print("Max IG was: ", max_ig)

	print("Learning the DT")

	full_dict = {}
	np_df = np.array(df)
	part2_train.create_dag(np_df, child, full_dict, 1, MAX_DEPTH, out)

	model = full_dict

	train_acc = []
	test_acc = []
	depth = 0

	f_label = open("dataset for part 2/words.txt", "r")
	mapping = []
	while(True):
		line = f_label.readline()
		if not line:
			break
		mapping.append(line)

	f_label.close()

	for i in range(1, MAX_DEPTH + 1):
		print("############################")
		print("Inference upto depth: ", i)
		print("############################")
		pretty_print.pretty_print_part(model, "3566", 1, mapping)

		children = list(df_test.columns)
		children = [str(x) for x in children]
		out = children[-1]
		children = children[:-1]
		child = children[0]

		total_num = df[int(out)].count()
		acc = 0
		for j in range(df[int(out)].count()):
			row_here = df.loc[j]
			result = part2_test.infer(row_here, copy(model), out, req_depth = i)
			if (result == list(row_here)[-1]):
				acc+=1

		print("Train accuracy: "+str(float(acc)/total_num))

		acc = 0
		total_num = df_test[int(out)].count()
		for j in range(df_test[int(out)].count()):
			row_here = df_test.loc[j]
			result = part2_test.infer(row_here, copy(model), out, req_depth = i)
			if (result == list(row_here)[-1]):
				acc+=1

		test_acc.append(float(acc)/total_num)
		print("Test accuracy: "+str(float(acc)/total_num))
		print()

	train_acc_scikit, test_acc_scikit = part2_scikit.dtc(df, df_test, "entropy", MAX_DEPTH)

import pandas as pd
import numpy as np
import json
import pudb
from copy import copy

# pu.db
def infer(row, model, out, req_depth = None, depth = 1):
	while(1):
		# print("here")
		choice = [x for x in model][0]
		if choice == "__result__":
			choice = [x for x in model][1]

		if req_depth != None and depth > req_depth:
			return model["__result__"]
		# pu.db
		actual_value = row[int(choice)]
		# print("model: "+str(model))
		# print("choice: "+str(choice))
		model = model[choice]
		# pu.db
		# print()
		if choice == out:
			return model
		if (str(actual_value) not in model) or actual_value == out:
			pu.db
			return model[out]
		model = model[str(actual_value)]
		depth += 1


if __name__ == "__main__":
	print("Loading data")
	df = create_df()
	df_test = create_df_test()
	f = open("model_part_2.json", "r")
	model = json.load(f)
	f.close()

	children = list(df.columns)
	children = [str(x) for x in children]
	out = children[-1]
	children = children[:-1]
	child = children[0]

	acc = 0
	total_num = df[int(out)].count()
	for i in range(df[int(out)].count()):
		row_here = df.loc[i]
		result = infer(row_here, copy(model))
		if (result == list(row_here)[-1]):
			acc+=1

	print("Training accuracy: "+str(float(acc)/total_num))


	children = list(df_test.columns)
	children = [str(x) for x in children]
	out = children[-1]
	children = children[:-1]
	child = children[0]

	acc = 0
	total_num = df_test[int(out)].count()
	for i in range(df_test[int(out)].count()):
		row_here = df_test.loc[i]
		result = infer(row_here, copy(model))
		if (result == list(row_here)[-1]):
			acc+=1

	print("Test accuracy: "+str(float(acc)/total_num))
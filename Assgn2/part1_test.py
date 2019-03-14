import pandas as pd
import numpy as np
import json
import pudb
from copy import copy

f = open("model_part_1a.json", "r")
model = json.load(f)
f.close()

df = pd.read_csv("dataset for part 1 - Test Data.csv")

acc = 0
total_num = df["profitable"].count()

# pu.db


def infer(row, model):
	# print(row)
	while(1):
		# print("here")
		choice = [x for x in model][0]
		actual_value = row[choice]
		model = model[choice]
		# pu.db
		if choice == "profitable":
			return model
		if (str(actual_value) not in model) or actual_value == "profitable":
			# pu.db
			return model["profitable"]
		model = model[str(actual_value)]




for i in range(df["profitable"].count()):
	row_here = df.loc[i]
	result = infer(row_here, copy(model))
	# print(result)
	if (result == list(row_here)[-1]):
		acc+=1

print("Test accuracy: "+str(float(acc)/total_num))

df = pd.read_csv("dataset for part 1 - Training Data.csv")

acc = 0
total_num = df["profitable"].count()



for i in range(df["profitable"].count()):
	row_here = df.loc[i]
	result = infer(row_here, copy(model))
	# print(result)
	if (result == list(row_here)[-1]):
		acc+=1

print("Training accuracy: "+str(float(acc)/total_num))

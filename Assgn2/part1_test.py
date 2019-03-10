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
	while(1):
		# print("here")
		choice = [x for x in model][0]
		actual_value = row[choice]
		model = model[choice]
		# pu.db
		if (str(actual_value) not in model) or actual_value == "profitable":
			# pu.db
			return model["profitable"]
		if choice == "profitable":
			return model
		model = model[str(actual_value)]




for i in range(df["profitable"].count()):
	row_here = df.loc[0]
	result = infer(row_here, copy(model))
	if (result == list(row_here)[-1]):
		acc+=1

print("Accuracy: "+str(float(acc)/total_num))
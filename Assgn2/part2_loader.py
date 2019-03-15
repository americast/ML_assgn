import pandas as pd
import numpy as np


def create_df(data="train"):
	f = open("dataset for part 2/"+data+"data.txt", "r")
	f_label = open("dataset for part 2/"+data+"label.txt", "r")
	old_docID, old_wordID = 1, 1
	old_flag = False
	if data == "train":
		df = np.zeros((1060, 3567), dtype = np.uint64)
	else:
		df = np.zeros((707, 3567), dtype = np.uint64)

	count = -1
	while(1):
		count += 1
		# print(count)
		# words = [0 for x in range(3566)]
		if old_flag:
			df[count, old_wordID - 1] = 1
		while(1):
			line = f.readline()
			if not line:
				break
			line = line.split()
			docID, wordID = int(line[0]), int(line[1])
			if (docID != old_docID):
				old_wordID = wordID
				old_docID = docID
				old_flag = True
				break
			df[count, wordID - 1] = 1

		label = int(f_label.readline())

		# words.append(label)
		df[count, -1] = label
		# df = df.append([pd.Series(words)])
		if not line:
			break
		# print("Appended to df")
		# pu.db


	df = pd.DataFrame(df)
	f.close()
	f_label.close()
	return df


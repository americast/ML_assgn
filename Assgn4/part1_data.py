from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
import numpy as np
from numpy import random
import pudb

stemmer = PorterStemmer()

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

tokenizer = RegexpTokenizer(r'\w+')

res = []
docs = []
all_words = {}

def data_loader(X, y, batch_size):
    mapIndexPosition = list(zip(X, y))
    random.shuffle(mapIndexPosition)
    X_shuffled, y_shuffled = zip(*mapIndexPosition)
    # pu.db
    mini_batches = [[list(X_shuffled[i:i+batch_size]), list(y_shuffled[i:i+batch_size])] for i in range(0, len(y), batch_size)]
    return mini_batches

def get_data():
	f = open("Assignment_4_data.txt", "r")

	while(1):
		line = f.readline()
		if not line:
			break
		num_chars = len(line.split()[0])
		type_ = line[:num_chars]
		text = line[num_chars:]
		tokens = tokenizer.tokenize(text.lower())

		tokens = [stemmer.stem(w) for w in tokens if not w in stop_words] 

		if (type_=="spam"):
			res.append(1)
		else:
			res.append(0)

		docs.append(tokens)

		for word in tokens:
			if word not in all_words:
				all_words[word] = 1
			else:
				all_words[word] += 1

	all_words_list = []
	for each in all_words:
		all_words_list.append([each, all_words[each]])

	all_words_list.sort(reverse = True, key = lambda x: x[1])
	all_words_list = all_words_list[:500]
	all_words_list = [x[0] for x in all_words_list]

	indices = {}
	count = 0

	for each in all_words_list:
		indices[each] = count
		count+=1

	docs_sparse = []

	for each in docs:
		# print()
		each_sparse = [0 for x in range(len(all_words_list))]
		for e in each:
			if e in all_words_list:
				# print(indices[e])
				each_sparse[indices[e]] = 1
		docs_sparse.append(np.reshape(np.array(each_sparse), (500, 1)))

	return res, docs_sparse

if __name__ == "__main__":
	res, docs_sparse = get_data()
	mini_batches = data_loader(docs_sparse, res, 1)

	pu.db

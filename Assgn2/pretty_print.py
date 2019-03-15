import json
import pudb

def pretty_print_part(model, out, depth = 1, mapping = None):
	choice = [x for x in model][0]
	if choice == out:
		print(" : "+str(model[choice]))
		return
	print()
	if (depth > 2):
		for i in range(1, depth - 1):
			print("\t", end="")
	if (depth > 1):
		print("| ", end="")

	model_here = model[choice]


	for each in model_here:
		# pu.db
		if mapping == None:
			print(choice+" = "+each, end="")
		else:
			print(mapping[int(choice)][:-1]+" = "+str(each), end="")

		pretty_print_part(model_here[each], out, depth + 1, mapping)

if __name__ == "__main__":
	f = open("model_part_2.json", "r")
	model = json.load(f)
	f.close()

	f_label = open("dataset for part 2/words.txt", "r")
	mapping = []
	while(True):
		line = f_label.readline()
		if not line:
			break
		mapping.append(line)

	pretty_print_part(model, "3566", mapping = mapping)	
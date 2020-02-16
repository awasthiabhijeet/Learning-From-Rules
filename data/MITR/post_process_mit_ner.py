import os,sys,pickle

label2class = {0: 'O',
               1: 'Location', 2: 'Hours',
               3: 'Amenity', 4: 'Price',
               5: 'Cuisine', 6: 'Dish',
               7: 'Restaurant_Name', 8: 'Rating'}

test_labels = "restauranttest.bio"
infer_dir = sys.argv[1]
BIO = sys.argv[2]
print(infer_dir)

infer_labels = os.path.join(infer_dir,"infer_f.p")
output = os.path.join(infer_dir,"infer_output_{}_BIO.txt".format(BIO))

if BIO == "True":
	BIO = True
elif BIO == "False":
	BIO = False
else:
	print("Wrong BIO")
	exit(1)

infer_labels_f = open(infer_labels,"rb")
infer_labels = pickle.load(infer_labels_f)
infer_labels = pickle.load(infer_labels_f)
infer_labels = pickle.load(infer_labels_f)
infer_labels = pickle.load(infer_labels_f)
infer_labels_f.close()
print(len(infer_labels))
#exit(1)


with open(test_labels,"r") as test_f, open(output,"w") as output_f:
	count = 0
	prev_pred = None
	prev_docstart = False
	count_DOCSTART = 0
	for line in test_f:
		line = line.strip().split()
		if line and line[0] == "-DOCSTART-":
			count_DOCSTART +=1
			if count_DOCSTART!=1:
				count +=1
			prev_docstart = True
			continue

		if prev_docstart:
			prev_docstart = False
			continue

		if not line:
			output_f.write('\n')
			prev_pred=None
		else:
			token = line[1]
			gold = line[0]
			pred = infer_labels[count]
			pred = label2class[pred]
			if BIO:
				if pred == 'O':
					pass
				else:
					for cat in label2class.values():
						if pred == cat:
							if prev_pred in ['B-'+pred,'I-'+pred]:
								pred = 'I-'+pred
							else:
								pred = 'B-'+pred
						else:
							continue
			else:
				if gold == 'O':
					pass
				else:
					gold=gold[2:]
					#print(token)

			output_f.write("{}\t{}\t{}\n".format(token,gold,pred))
			count += 1
			prev_pred = pred

assert count == len(infer_labels)



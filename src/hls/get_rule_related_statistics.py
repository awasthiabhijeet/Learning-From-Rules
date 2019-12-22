# gets majority voting accuracy from validation_processed.p and test_processed.p (or any pickle of similar format)
# for validation and test data respectively

import pickle
import sys,os
from collections import Counter
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support

def get_majority_vote(l, num_classes, default_class=None):
	# if no rule fire, 
	#	default label is the output if provided, 
	#   else a random label chosen uniformly
	# else
	#	majority label is the output
	# 	ties between multiple majority classes are broken arbitarily
	result = []
	for i in range(len(l)):
		c = Counter(l[i]).most_common()
		if c[0][0] == num_classes and len(c)==1:
			if (default_class is None) or (default_class == "None"):
				result.append(random.randint(0,num_classes-1))
			else:
				result.append(int(default_class))
		else:
			c = [item for item in c if item[0]!=num_classes]
			majority_freq = c[0][1]
			freq_classes = [item[0] for item in c if item[1]==majority_freq]
			result.append(random.choice(freq_classes))
	return np.array(result)

path_dir = sys.argv[1] #path to data strored in form of pickles
num_classes = int(sys.argv[2]) #num of classes to be predicted (depends on the dataset)
default_class = sys.argv[3] #default class (usually the most frequent class) can also be "None"

validation_pickle = open(os.path.join(path_dir,"validation_processed.p"),"rb")
validation_x=pickle.load(validation_pickle)
validation_l=pickle.load(validation_pickle)
validation_m=pickle.load(validation_pickle)
validation_L=pickle.load(validation_pickle) #true labels
validation_d=pickle.load(validation_pickle)

majority = get_majority_vote(validation_l,num_classes,default_class) #majority voted predictions
accuracy = np.sum(np.equal(majority,validation_L).astype(np.float))/len(validation_L)

precision, recall, f1_score, support = precision_recall_fscore_support(validation_L, majority)

print("Accuracy of majority voting on validation data: ", accuracy)
print("Precision of majority voting on validation data: ", precision)
print("Recall of majority voting on validation data: ", recall)
print("f1_score of majority voting on validation data: ", f1_score)
print("support of majority voting on validation data: ", support)

U_pickle = open(os.path.join(path_dir,"U_processed.p"),"rb")
U_x=pickle.load(U_pickle)
U_l=pickle.load(U_pickle)
U_m=pickle.load(U_pickle)
U_L=pickle.load(U_pickle)
U_d=pickle.load(U_pickle)
fired_U_idx = [i for i,item in enumerate(U_m) if np.sum(item)!=0] # instance indices on which 
																  # atleast 1 rule fired

print("Number of rules: ", U_l.shape[1])
print("Size of Validation: ",len(validation_x))
print("Size of U: ",len(U_x))
print("Size of fired U: ", len(fired_U_idx))
print("Cover percentage: ",len(fired_U_idx)/len(U_x))


d_pickle = open(os.path.join(path_dir,"d_processed.p"),"rb")
d_x=pickle.load(d_pickle)
d_l=pickle.load(d_pickle)
d_m=pickle.load(d_pickle)
d_L=pickle.load(d_pickle)
d_d=pickle.load(d_pickle)
print("Size of d: ",len(d_x))

test_pickle = open(os.path.join(path_dir,"test_processed.p"),"rb")
test_x=pickle.load(test_pickle)
test_l=pickle.load(test_pickle)
test_m=pickle.load(test_pickle)
test_L=pickle.load(test_pickle)
test_d=pickle.load(test_pickle)
test_fired_idx = [i for i,item in enumerate(test_m) if sum(item)>0]

majority = get_majority_vote(test_l,num_classes,default_class)
#dump majority preds for test data in a text file for external evaluation (if needed)
with open(os.path.join(path_dir,"majority_voting_preds.txt"),"w") as pred_file:
	for item in majority :
		pred_file.write(str(item)+"\n")


accuracy = np.sum(np.equal(majority,test_L).astype(np.float))/len(test_L)

precision, recall, f1_score, support = precision_recall_fscore_support(test_L, majority)

print("Accuracy of majority voting on test data: ", accuracy)
print("Precision of majority voting on test data: ", precision)
print("Recall of majority voting on test data: ", recall)
print("f1_score of majority voting on test data: ", f1_score)
print("support of majority voting on test data: ", support)

print("size of test: ",len(test_x))
print("size of fired_test: ",len(test_fired_idx))

def get_rule_precision(l,L,m):
	#micro_p : correct_rule_firings/total_rule_firings (micro_precision)
	#macro_p : average of individual precision of rules having non-zero support
	#rule_wise_precision : individual precision of rules

	L = L.reshape([L.shape[0],1])
	comp = np.equal(l,L).astype(np.float)
	comp = comp * m
	comp = np.sum(comp,0)
	support = np.sum(m,0)
	micro_p = np.sum(comp)/np.sum(support)
	macro_p = comp/(support + 1e-25)
	supported_rules = [idx for idx,support_val in enumerate(support) if support_val>0]
	macro_p = macro_p[supported_rules]
	macro_p = np.mean(macro_p)
	rule_wise_precision = comp/(support + 1e-25)
	return micro_p,macro_p,rule_wise_precision

micro_p,macro_p,rule_wise_precision = get_rule_precision(test_l,test_L,test_m)
print("Micro Precision of rules on test data: ",micro_p)

def get_conflict_rule_cov_rule_per_inst(l,m):
	rule_cov = np.mean(np.sum(m,0))
	rules_per_inst = np.mean(np.sum(m,1))
	conflicts = 0
	for i in range(len(l)):
		uniques = np.unique(l[i])
		if len(uniques) >=3:
			conflicts +=1
		else:
			if (len(uniques)==2 and num_classes in uniques) or len(uniques)==1:
				continue
			else:
				conflicts +=1
	avg_conflicts = conflicts/m.shape[0]
	return avg_conflicts, rule_cov, rules_per_inst

conflicts,rule_cov,rules_per_inst = get_conflict_rule_cov_rule_per_inst(U_l[fired_U_idx],U_m[fired_U_idx])
print("Conflict rate in U: ",conflicts)
print("Average num of instances covered by any rule in U: ",rule_cov)
print("Average rules firing on an instance in U: ", rules_per_inst)




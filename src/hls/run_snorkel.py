# using snorkel for obtaining soft labels 
# trains a snorkel model based on rule firing matrices (l and m)

import os,sys
import pickle
from snorkel.labeling import LabelModel
from snorkel.labeling import MajorityLabelVoter
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from snorkel_utils import conv_l_to_lsnork

'''
conv_l_to_lsnork: 

in snorkel convention is
if a rule does not cover an instance assign it label -1
we follow the convention where we assign the label num_classes instead of -1
valid class labels range from {0,1,...num_classes-1}
conv_l_to_lsnork:  converts l in our format to snorkel's format
'''

def compute_accuracy(support, recall):
	return np.sum(support * recall) / np.sum(support)


path_dir = sys.argv[1]  # path where data pickles are stored
num_classes = int(sys.argv[2]) # number of classes (depends on the dataset)
default_class = sys.argv[3]  # default class (can be provided as None if no default class exists)
							 # usually the most frequent class in a dataset with high imbalance

if default_class=="None":
	default_class=None
else:
	default_class=int(default_class) 

# snorkel's majority voting model
majority_model = MajorityLabelVoter(cardinality=num_classes)

# load unlabeled data
U_file = open(os.path.join(path_dir,"U_processed.p"),"rb")
U_x = pickle.load(U_file)
U_l = pickle.load(U_file)
U_m = pickle.load(U_file)
U_L = pickle.load(U_file)
U_d = pickle.load(U_file)
U_lsnork = conv_l_to_lsnork(U_l,U_m)
#indices of instances where atleast one rule fired
U_fired_idx = [i for i,item in enumerate(U_m) if sum(item)>0] 

# load test data
test_file = open(os.path.join(path_dir,"test_processed.p"),"rb")
test_x = pickle.load(test_file)
test_l = pickle.load(test_file)
test_m = pickle.load(test_file)
test_L = pickle.load(test_file)
test_d = pickle.load(test_file)
test_lsnork = conv_l_to_lsnork(test_l,test_m)
#indices of instances where atleast one rule fired
test_fired_idx = [i for i,item in enumerate(test_m) if sum(item)>0]
#indices of instances where no rule fired
test_unfired_idx = [i for i,item in enumerate(test_m) if sum(item)==0]
targets_test = test_L[test_fired_idx]

#majority voting using snorkel's majority voting model
maj_preds_test = majority_model.predict(L=test_lsnork[test_fired_idx])
maj_precision_test, maj_recall_test, maj_f1_score_test, maj_support_test = precision_recall_fscore_support(targets_test, maj_preds_test)
maj_accuracy_test = compute_accuracy(maj_support_test, maj_recall_test)

print("precision on *** RULE COVERD TEST SET ***   of MAJORITY VOTING: {}".format(maj_precision_test))
print("recall on *** RULE COVERED TEST SET ***  of MAJORITY VOTING: {}".format(maj_recall_test))
print("f1_score on *** RULE COVERED TEST SET *** of MAJORITY VOTING: {}".format(maj_f1_score_test))
print("support on *** RULE COVERED TEST SET ***  of MAJORITY VOTING: {}".format(maj_support_test))
print("accuracy on *** RULE COVERED TEST SET ***   of MAJORITY VOTING: {}".format(maj_accuracy_test))


#Now train snorkels label model
print("Training Snorkel's LabelModel")
label_model = LabelModel(cardinality=num_classes, verbose=True)
label_model.fit(L_train=U_lsnork, n_epochs=1000, lr=0.001, log_freq=100, seed=123)
label_model.save(os.path.join(path_dir,"saved_label_model"))



snork_preds_test = label_model.predict(L=test_lsnork[test_fired_idx])
snork_precision_test, snork_recall_test, snork_f1_score_test, snork_support_test = precision_recall_fscore_support(targets_test, snork_preds_test)
snork_accuracy_test = compute_accuracy(snork_support_test, snork_recall_test)
print("precision on *** RULE COVERED TEST SET *** of SNORKEL VOTING: {}".format(snork_precision_test))
print("recall on *** RULE COVERED TEST SET *** of SNORKEL VOTING: {}".format(snork_recall_test))
print("f1_score on *** RULE COVERED TEST SET *** of SNORKEL VOTING: {}".format(snork_f1_score_test))
print("support on *** RULE COVERED TEST SET *** of SNORKEL VOTING: {}".format(snork_support_test))
print("accuracy on *** RULE COVERED TEST SET *** of SNORKEL VOTING: {}".format(snork_accuracy_test))
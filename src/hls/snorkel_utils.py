import numpy as np

def conv_l_to_lsnork(l,m):
	'''
	in snorkel convention is
	if a rule does not cover an instance assign it label -1
	we follow the convention where we assign the label num_classes instead of -1
	valid class labels range from {0,1,...num_classes-1}
	conv_l_to_lsnork:  converts l in our format to snorkel's format
	'''
	lsnork = l*m + -1*(1-m)
	return lsnork.astype(np.int)
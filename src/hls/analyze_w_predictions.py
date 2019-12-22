# Diagnostics: Effectiveness of learning true coverage via Pjφ
# analysis of proabilities (weights)
# output by the w network (rule network)
# for each instance, w network provides a probabilitiy for each rule
# whether the rule correctly generalizes to the instance or not

# used to compare the original precision of rules
# and the denosied rule-precision from learned w network ( Pjφ )

import os,pickle
import numpy as np

def analyze_w_predictions(x,l,m,L,d,weights,probs,rule_classes):
	num_classes = probs.shape[1]
	new_m = convert_weights_to_m(weights) * m
	new_l = convert_m_to_l(new_m,rule_classes,num_classes)
	o_micro,o_marco_p,o_rp = get_rule_precision(l,L,m)
	n_mirco,new_macro_p,n_rp = get_rule_precision(new_l,L,new_m)
	print("old micro precision: ", o_micro)
	print("new micro precision: ", n_mirco)
	print("old rule firings: ", np.sum(m))
	print("new rule firings: ", np.sum(new_m))
	print("old rule coverage: ", len([i for i in m if sum(i) > 0]))
	print("new rule firings: ", len([i for i in new_m if sum(i) > 0]))

def convert_weights_to_m(weights):
	new_m = weights > 0.5
	new_m = new_m.astype(np.int32)
	return new_m

def convert_m_to_l(m,rule_classes,num_classes):
	rule_classes = np.array([rule_classes]*m.shape[0])
	l = m * rule_classes + (1-m)*num_classes
	return l

def get_rule_precision(l,L,m):
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
	return micro_p,macro_p,comp/(support + 1e-25)
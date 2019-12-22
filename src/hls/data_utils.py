import pickle
import sys
import numpy as np

from utils import *

def dump_labels_to_file(save_filename, x, l, m, L, d, weights=None, f_d_U_probs=None, rule_classes=None):
    save_file = open(save_filename, 'wb')
    pickle.dump(x, save_file)
    pickle.dump(l, save_file)
    pickle.dump(m, save_file)
    pickle.dump(L, save_file)
    pickle.dump(d, save_file)

    if not weights is None:
        pickle.dump(weights, save_file)

    if not f_d_U_probs is None:
        pickle.dump(f_d_U_probs, save_file)

    if not rule_classes is None:
        pickle.dump(rule_classes,save_file)

    save_file.close()

def load_from_pickle_with_per_class_sampling_factor(fname, per_class_sampling_factor):
    with open(fname, 'rb') as f:
        x = pickle.load(f)
        l = pickle.load(f)
        m = pickle.load(f)
        L = pickle.load(f)
        d = np.squeeze(pickle.load(f))

    x1 = []
    l1 = []
    m1 = []
    L1 = []
    d1 = []
    for xx, ll, mm, LL, dd in zip(x, l, m, L, d):
        for i in range(per_class_sampling_factor[LL]):
            x1.append(xx)
            l1.append(ll)
            m1.append(mm)
            L1.append(LL)
            d1.append(dd)

    x1 = np.array(x1)
    l1 = np.array(l1)
    m1 = np.array(m1)
    L1 = np.array(L1)
    d1 = np.array(d1)

    return x1, l1, m1, L1, d1


def combine_d_covered_U_pickles(d_name, infer_U_name, out_name, d_sampling_factor, U_sampling_factor):
    
    #d_sampling_factor = np.array(d_sampling_factor)
    #U_sampling_factor = np.array(U_sampling_factor)

    d_x, d_l, d_m, d_L, d_d = load_from_pickle_with_per_class_sampling_factor(d_name, d_sampling_factor)
    U_x, U_l, U_m, U_L, U_d = load_from_pickle_with_per_class_sampling_factor(infer_U_name, U_sampling_factor)

    x = np.concatenate((d_x, U_x))
    l = np.concatenate((d_l, U_l))
    m = np.concatenate((d_m, U_m))
    L = np.concatenate((d_L, U_L))
    #print(d_d.shape)
    #print(U_d.shape)
    d = np.concatenate((d_d, U_d))

    with open(out_name, 'wb') as out_file:
        pickle.dump(x, out_file)
        pickle.dump(l, out_file)
        pickle.dump(m, out_file)
        pickle.dump(L, out_file)
        pickle.dump(d, out_file)
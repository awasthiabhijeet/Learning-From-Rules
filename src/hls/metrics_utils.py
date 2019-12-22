import numpy as np

def compute_accuracy(support, recall):
    return np.sum(support * recall) / np.sum(support)


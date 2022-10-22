import numpy as np

def mean(data):
    return np.mean(data)

def std(data):
    return np.std(data)

def mse(y, y_pred):
    total = 0
    for gt, pd in zip(y, y_pred):
        total += (gt - pd) ** 2
    return total / len(y)
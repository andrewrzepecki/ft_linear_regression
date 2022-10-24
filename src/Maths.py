import math

def mean(data):
    return sum(data) / len(data)

def std(data):
    xm = mean(data)
    std = 0
    for x in data:
        std += (x - xm)**2

    std = std / len(data)
    std = math.sqrt(std)
    return std

def mse(y, y_pred):
    total = 0
    for gt, pd in zip(y, y_pred):
        total += (gt - pd) ** 2
    return total / len(y)
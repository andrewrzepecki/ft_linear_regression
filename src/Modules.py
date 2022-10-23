import numpy as np
try:
    from Maths import std, mean
except Exception:
    from src.Maths import std, mean

class Module():

    def __init__(self, dim : int = 1, trainable : bool = True):
        self.trainable = trainable
        self.dim = dim
        pass
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return x

class Normalize(Module):

    def __init__(self, dim : int = 1, trainable : bool = False):
        super().__init__()
        self.min = np.ones(dim)
        self.max = np.ones(dim)
        self.dim = dim
        self.trainable = trainable

    def forward(self, x : list):
        y = []
        for X, MIN, MAX in zip(x, self.min, self.max):
            y.append((X - MIN) / (MAX - MIN))
        return y

    def fit(self, x):
        self.dim = len(x)
        self.min = np.ones(self.dim)
        self.max = np.ones(self.dim)
        for i, X in enumerate(x):
            self.min[i] = min(X)
            self.max[i] = max(X)

class Standardize(Module):
    
    def __init__(self, dim : int = 1, trainable : bool = False):
        super().__init__()
        self.means = np.ones(dim)
        self.stds = np.ones(dim)
        self.dim = dim
        self.trainable = trainable

    def forward(self, x : list):
        y = []
        for X, M, STD in zip(x, self.means, self.stds):
            y.append((X - M) / STD)
        return y

    def fit(self, x):
        self.dim = len(x)
        self.means = np.ones(self.dim)
        self.stds = np.ones(self.dim)
        for i, X in enumerate(x):
            self.means[i] = mean(X)
            self.stds[i] = std(X)


class Linear(Module):

    def __init__(self, dim : int = 1, trainable : bool = True):
        super().__init__()
        # a
        self.coefficients = np.zeros(dim)
        self.dim = len(self.coefficients)
        # b
        self.intercept = 0
        self.trainable = trainable

    def forward(self, x):
        y = 0
        for X, A in zip(x, self.coefficients):
            y += A * X
        return y + self.intercept

    def fit(self, x, y, lr):
        
        if self.dim != len(x[0]):
            self.intercept = 0
            self.dim = len(x[0])
            self.coefficients = np.zeros(self.dim)
        
        coefficients_gradient = np.zeros(self.dim)
        intercept_gradient = 0
        n = len(x)
        
        for i in range(n):
            intercept_gradient += self.forward(x[i]) - y[i]
            for j in range(self.dim):
                coefficients_gradient[j] += (self.forward(x[i]) - y[i]) * x[i][j]
        
        self.intercept = self.intercept - intercept_gradient * (1/n) * lr
        for i in range(self.dim):
            self.coefficients[i] = self.coefficients[i] - coefficients_gradient[i] * (1/n) * lr
            

MODULE_MAP = {
    'Standardize' : Standardize,
    'Linear' : Linear,
    'Normalize' : Normalize
}
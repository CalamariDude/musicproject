import numpy as np 
from sklearn.datasets import load_iris

class neural_network:
    def activation(self, y):
        result = 1/(1+np.exp(-y))
        return result
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 5

    


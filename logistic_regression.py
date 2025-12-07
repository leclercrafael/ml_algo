import numpy as np
import scipy as sp
import matplotlib.pylab as plt

from linear_regression import myLinearRegression

class myLogisticRegression:

    def __init__(self, learning_rate : float = 0.01, n_iter : int = 10000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def sigmoid(z : float) -> float:
        return 1/(1+np.exp(-z))
    
    def loss_function(y : np.array, y_pred : np.array) -> float:
        return -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    
    def dloss_db(self, x : np.array, y : np.array, w : np.array, b : float) -> float : 
        N = len(x)
        y_pred = self.prediction(x, w, b)
        return 2*np.sum(y_pred - y)/N
    
    

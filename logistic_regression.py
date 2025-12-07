import numpy as np
import scipy as sp
import matplotlib.pylab as plt

from linear_regression import myLinearRegression

class myLogisticRegression:

    def __init__(self, learning_rate : float = 0.01, n_iter : int = 10000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def linear(self, x : np.array, w : np.array, b : float) -> np.array :
        return (np.dot(x,w) + b)

    def sigmoid(z : float) -> float:
        return 1/(1+np.exp(-z))
    
    def loss_function(y : np.array, y_pred : np.array) -> float:
        return -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    
    def dloss_db(self, x : np.array, y : np.array, w : np.array, b : float) -> float : 
        N = len(x)
        y_pred = self.prediction(x, w, b)
        return 2*np.sum(y_pred - y)/N
    
    def dloss_dw(self, x : np.array, y : np.array, w : np.array, b : float) -> np.array :
        N = len(x)
        y_pred = self.prediction(x, w, b)
        return 2*np.dot(x.T, (y_pred-y))/N
    
    def fit(self, X : np.array):

        X = np.array(X)

        self.cost_history = []
         
        self.x = X[:,:-1]
        self.y = X[:,-1]

        self.w = np.array([1 for i in range(len(self.x[0]))])
        self.b = 0

        for i in range (self.n_iter):

            dmse_db = self.dmse_db(self.x, self.y, self.w, self.b)
            dmse_dw = self.dmse_dw(self.x, self.y, self.w, self.b)

            self.b = self.b -self.learning_rate*dmse_db
            self.w = self.w -self.learning_rate*dmse_dw

            current_cost = self.mse(self.x, self.y, self.w, self.b)
            self.cost_history.append(current_cost)

    def plot_learning_curve(self):
        
        plt.plot(self.cost_history)
        plt.title("Évolution de la fonction de coût (MSE)")
        plt.xlabel("Itérations")
        plt.ylabel("Coût")
        plt.show()

    @property

    def coef(self) -> None :
        print (f'Weights w : {self.w}')
        print (f'Biais b : {self.b}')

    @property

    def final_mse(self):
        return self.mse(self.x, self.y, self.w, self.b)


            

        



        


        




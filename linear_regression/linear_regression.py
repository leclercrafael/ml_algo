import numpy as np
import scipy
import matplotlib.pyplot as plt

class myLinearRegression():

    def __init__(self):
        #self.learning_rate = learning_rate
        #self.n_iter = n_iter
        pass

    def prediction(self, x : np.array, w : np.array, b : float) -> np.array :
        return (np.dot(x,w) + b)

    def mse(self, x : np.array, y: np.array, w : np.array, b : float) -> float:
        y_pred = self.prediction(x, w, b)
        return np.mean((y - y_pred)**2)
    
    def dmse_db(self, x : np.array, y : np.array, w : np.array, b : float) -> float : 
        N = len(x)
        y_pred = self.prediction(x, w, b)
        return 2*np.sum(y_pred - y)/N
    
    def dmse_dw(self, x : np.array, y : np.array, w : np.array, b : float) -> np.array :
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

        N_ITER = 1000

        for i in range (N_ITER):

            dmse_db = self.dmse_db(self.x, self.y, self.w, self.b)
            dmse_dw = self.dmse_dw(self.x, self.y, self.w, self.b)

            self.b = self.b -0.01*dmse_db
            self.w = self.w -0.01*dmse_dw

            current_cost = self.mse(self.x, self.y, self.w, self.b)
            self.cost_history.append(current_cost)

    def plot_learning_curve(self):
        
        plt.plot(self.cost_history)
        plt.title("Évolution de la fonction de coût (MSE)")
        plt.xlabel("Itérations")
        plt.ylabel("Coût")
        plt.show()

    @attr


            

        



        


        


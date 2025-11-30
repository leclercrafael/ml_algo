import numpy as np
import scipy

class LinearRegression():

    def __init__(self):
        #self.learning_rate = learning_rate
        #self.n_iter = n_iter
        pass

    def prediction(self, x : np.array, w : np.array, b : float) -> np.array :
        return (np.dot(x,w) + b)

    def mse(self, x : np.array, y: np.array, w : np.array, b : float) -> float:
        N = len(x)
        sum = 0
        for i in range(N): 
            sum += (y[i] - self.prediction(x, w, b))**2
        return sum/N
    
    def dmse_db(self, x : np.array, y : np.array, w : np.array, b : float) -> np.float : 
        sum =0
        N = len(x)
        for i in range(N):
            sum+=(self.prediction(x,w,b)-y[i])
        return 2*sum/N
    
    def dmse_dw(self, x : np.array, y : np.array, w : np.array, b : float) -> np.array :
        N = len(x)
        sum = np.zeros(N)
        for i in range(N):
            sum +=(self.prediction(x,w,b)-y[i])* np.array(x[i])
        return 2*sum/N

    def fit(self, X : np.array):
         
        self.x = X[:,:-1]
        self.y = X[:,-1]

        self.w = np.array([1 for i in range(len(x[0]))])
        self.b = 0

        N_ITER = 1000

        for i in range (N_ITER):

            dmse_db = dmse_db(self.x, self.y, self.w, self.b)
            dmse_dw = dmse_dw(self.x, self.y, self.w, self.b)

            self.b = self.b -0.1*dmse_db
            self.w = self.w -0.1*dmse_dw

    def transform():
        pass


            

        



        


        


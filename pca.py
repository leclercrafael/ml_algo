import numpy as np
import scipy
import matplotlib.pyplot as plt


class myPCA:

    def __init__(self, n_components : int) -> None:
        self.n_components = n_components

    def fit(self, X : np.array) -> None:
        #On calcule la moyenne pour centrer les données
        self.mean = np.mean(X)
        
        #On centre les données puis on calcule la matrice de covariance
        X_centered = X - self.mean
        cov_matrix = np.dot(X_centered.T, X_centered)*(1/len(X))

        #On calcule les vecteurs propres et on les trie en fonction de la valeur propre décroissante
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues=eigenvalues[idxs]
        eigenvectors=eigenvectors[:,idxs]

        self.components = eigenvectors[:,:self.n_components]

    def transform(self, X : np.array) -> None:

        X_centered = X - self.mean

        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X : np.array):

        self.fit(X)
        return self.transform(X)









    



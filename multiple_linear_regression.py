import numpy as np
from sklearn.datasets import load_boston


boston = load_boston()

X_,y = boston.data,boston.target

X = np.zeros((X_.shape[0],X_.shape[1]+1))
X[:,:-1] = X_
X[:,-1] = 1
n_features = len(X[0])

weights = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,y))




def predict(X):
    
    return np.dot(weights,X.T)

def SSR(X,y):
    yHats = predict(X)
    return np.sum(np.power(yHats - y,2))

def SST(X,y):
    yMean = np.mean(y)
    return np.sum(np.power(y - yMean,2))

R_2 = 1 - SSR(X,y)/SST(X,y)
print('R_2',R_2)




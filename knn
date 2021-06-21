import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris,load_boston
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
Y = iris.target

x_train,x_test, y_train,y_test = train_test_split(X,Y,test_size=0.2)

def euc(p1,p2):
    return np.sqrt(np.sum(np.power(p1 - p2,2),axis=1))


p1 = np.array([7,4,3])
p2 = np.array([[17,6,2],[8,-2,4]])


class KNN:
    def __init__(self,k):
        self.k = k
    def fit(self,X_train,Y_train):
        self.x_train = X_train
        self.y_train = Y_train
        
    
    def predict(self,row):
        ds = euc(row,self.x_train)
        idxs_k_min_vs = ds.argsort()[:self.k]
        count = {}
        for i in idxs_k_min_vs:
            if self.y_train[i] not in count:
                count[self.y_train[i]] = 0
            count[self.y_train[i]] +=1
        predicted_lbl = np.max([k for k,v in count.items()])
        return predicted_lbl
        
if __name__  == '__main__':
    prediction_accuracy = []
    how_many_ks = 15
    for k in range(1,how_many_ks):
        knn = KNN(k)
        knn.fit(x_train,y_train)
        n_corrects = 0
        for idx in range(len(y_test)):
            n_corrects += 1 if knn.predict(x_test[idx]) == y_test[idx] else 0

        score = n_corrects/len(y_test)
        prediction_accuracy.append(score)

    plt.plot([x for x in range(1,how_many_ks)],prediction_accuracy)
    plt.show()

        

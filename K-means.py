import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
iris=pd.read_csv('iris.csv',header=None)
iris.columns=['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
X=iris[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
Y=iris['Species']
print(X.head())
def k_means(X,K):
    centroids_history=[]
    labels_history=[]
    rand_index=np.random.choice(X.shape[0],K)
    centroids=X[rand_index]
    centroids_history.append(centroids)
    while True:
        labels=np.argmin(cdist(X, centroids), axis=1)
        labels_history.append(labels)
        new_centroids=np.array([X[labels==i].mean(axis=0)for i in range(K)])
        centroids_history.append(new_centroids)
        if np.all(centroids==new_centroids):
            break
        centroids=new_centroids
    return centroids,labels,centroids_history,labels_history
X_mat=X.values
centroids,labels,centroids_history,label_history=k_means(X_mat,3)
print("Samples Labels")
print(labels)
plt.scatter(X['SepalLength'],X['SepalWidth'],c=labels,cmap='tab20b')
plt.title('Iris-Sepal Length vs Width-Clustered')
plt.show()

from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
data = pd.read_csv('./data.txt', sep=' ')
X = data[['calories', 'sodium', 'alcohol', 'cost']]
# X=StandardScaler().fit_transform(X)
''' n_clusters：聚类的K值 聚成几类 '''
km1 = KMeans(n_clusters=2).fit(X)
km2 = KMeans(n_clusters=3).fit(X)
data["km1"] = km1.labels_
data["km2"] = km2.labels_
centens1 = km1.cluster_centers_

colors = np.array(['red', 'blue', 'yellow', 'black', 'green'])
plt.scatter(data[['calories']], data[['alcohol']], s=100, alpha=1, c=colors[data["km1"]])
plt.scatter(centens1[0, 0], centens1[0, 2], linewidths=3, s=300, marker='+', c='black')
plt.scatter(centens1[1, 0], centens1[1, 2], linewidths=3, s=300, marker='+', c='black')
plt.show()
scoer = []
for i in range(2, 15):
    km = KMeans(n_clusters=i).fit(X)
    scoer.append(metrics.silhouette_score(X, km.labels_))
print(scoer)
''' eps :半径  min_samples：最小密度'''
dbscan = DBSCAN(min_samples=3, eps=15).fit(X)
print(metrics.silhouette_score(X, dbscan.labels_))
data["dbscan"] = km2.labels_
plt.scatter(data[['calories']], data[['alcohol']], s=100, alpha=1, c=colors[data["dbscan"]])
plt.show()

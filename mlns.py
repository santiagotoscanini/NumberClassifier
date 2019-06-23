import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import numpy as np
from sklearn.datasets.samples_generator import make_blobs

X,Y = make_blobs(n_samples = 500, centers=4)

plt.scatter(X[:,0], X[:,1],s=50)

plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

kmeans.fit(X)

y_means = kmeans.predict(X)

centers = kmeans.cluster_centers_
print(centers)

plt.scatter(X[:,0], X[:,1], c = y_means, cmap = 'viridis')
plt.scatter(centers[:,0], centers[:,1], c = 'black', s = 200, alpha = 0.5)
plt.show()
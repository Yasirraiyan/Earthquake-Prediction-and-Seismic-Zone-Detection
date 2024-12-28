import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# উদাহরণ ডেটা তৈরি করা
data = {
    'X': [2, 2, 8, 5, 7, 6, 1, 4],
    'Y': [10, 5, 4, 8, 5, 9, 2, 7]
}
df = pd.DataFrame(data)
plt.scatter(df['X'],df['Y'],color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('data points')
plt.show()
kmeans=KMeans(n_clusters=2)
kmeans.fit(df)
centers=kmeans.cluster_centers_
labels=kmeans.labels_
print("Cluster Centers:\n", centers)
print("Labels:\n", labels)
plt.scatter(df['X'], df['Y'], c=labels, cmap='rainbow')
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='X', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering')
plt.show()

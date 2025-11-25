import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from  sklearn.preprocessing import StandardScaler

''' customer segmentation based on how much income and how much they spend'''
df = pd.read_csv('dataset.csv')

''' basic check '''
# print(df.head())
# print(df.shape)
# print(df.describe())

Income = df['INCOME'].values
Spend = df['SPEND'].values
X = np.array(list(zip(Income, Spend)))
# print(x)

''' Finding best k value '''
wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)

# plt.plot(range(1,11), wcss, color="blue", marker="8")   # k=4 is optimal(both K and inertia looks low)
# plt.title(" Optimal K value :")
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# plt.show()

# sc = StandardScaler()
# X_scaled = sc.fit_transform(X)

''' Fitting a KMeans to dataset where k=4 '''
model = KMeans(n_clusters=4, random_state=0)
y_means = model.fit_predict(X)

''' Visualizing cluster for k=4 
1 - low income and low spend 
2 - High income & High spenders
3 - low income & high spenders
4 - High Income & Moderate Spenders '''

plt.figure(figsize=(8,6))
plt.scatter(X[y_means==0,0], X[y_means==0,1], s=50, c='brown', label="1-Low income & low spenders")
plt.scatter(X[y_means==1,0], X[y_means==1,1], s=50, c='blue', label="2-High income & High spenders")
plt.scatter(X[y_means==2,0], X[y_means==2,1], s=50, c='green', label="3-low income & high spenders")
plt.scatter(X[y_means==3,0], X[y_means==3,1], s=50, c='cyan', label="4-High Income & Moderate Spenders")

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, marker='s', c='red', label='Centroids')
plt.title(" Income Spent Analysis :")
plt.xlabel("Income")
plt.ylabel("spend")
plt.legend()
plt.show()
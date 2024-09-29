from sklearn.datasets import load_iris
iris_data = load_iris()
X = iris_data.data  # Features

#2.preprocessing
#no need to take care of missing data here
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)      #standardising data by formula x-mean/sigma

#pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)   # x_pca contains two columns now

#3. Applying kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_pca)                        # contains cluster labels

# 1. Silhouette Score
from sklearn.metrics import silhouette_score, davies_bouldin_score
silhouette_avg = silhouette_score(X_pca, y_kmeans)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# 2. Davies-Bouldin Index
davies_bouldin_avg = davies_bouldin_score(X_pca, y_kmeans)
print(f'Davies-Bouldin Index: {davies_bouldin_avg:.2f}')

# 3. Inertia
inertia = kmeans.inertia_
print(f'Inertia: {inertia:.2f}')

# Visualization
import matplotlib.pyplot as plt  
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_kmeans, palette='viridis', s=100)
plt.title('Clusters after K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

#the three clusters formed are most likely to be the three flowers in the iris datasheet.
'''Cluster Summary
Cluster 0 (Green): Tightly grouped likely representing a  group.
Cluster 1 (Purple): Elongated, suggesting a continuous variation or trend.
Cluster 2 (Yellow): Also elongated but in a different direction, indicating a distinct group or pattern.'''

# the data set is presented in a CSV file, in which each columns
# feature1–feature5 contain numeric features.


import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

try:
        df1 = pd.read_csv("dataset_A.csv")  
        df2 = pd.read_csv("dataset_B.csv")  
except FileNotFoundError:
    print("file is not found")
    exit()

parser = argparse.ArgumentParser()
parser.add_argument("--clusters", type=int, default=None, help="Number of clusters (optional)")
args = parser.parse_args()

expected_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
for col in expected_columns:
    if col not in df1.columns or col not in df2.columns:
        raise ValueError(f"There is not column: {col}")
    
df = pd.concat([df1, df2], ignore_index=True)


df = df.dropna(subset=expected_columns)  
for col in expected_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce') 

df = df.dropna(subset=expected_columns)  

features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

X = df[features].copy() 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

if args.clusters is not None:
    n_clusters = args.clusters
else:

    best_score = -1
    best_k = 2
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    n_clusters = best_k

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
df['cluster'] = cluster_labels  

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for c in range(n_clusters):
    plt.scatter(X_2d[cluster_labels == c, 0], X_2d[cluster_labels == c, 1], label=f"Cluster {c}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Clusters visualization")
plt.legend()
plt.show()
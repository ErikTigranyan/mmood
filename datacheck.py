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
<<<<<<< HEAD
parser.add_argument("--clusters", type=int, default=None, help="Number of clusters")
=======
parser.add_argument("--clusters", type=int, default=None, help="Number of clusters (optional)")
parser.add_argument("--output_csv", type=str, default="clustered_output.csv",
                    help="Path to save clustered dataset")
parser.add_argument("--output_plot", type=str, default="clusters_plot.png",
                    help="Path to save cluster visualization")
>>>>>>> 0f9020fb1de11da36784e92967be359d8f6a7007
args = parser.parse_args()

expected_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
for col in expected_columns:
    if col not in df1.columns or col not in df2.columns:
        raise ValueError(f"There is not column: {col}")
    
df = pd.concat([df1, df2], ignore_index=True)


df = df.dropna(subset=expected_columns)  
for col in expected_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce') 

df = df.dropna(subset=expected_columns)  

features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']

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
df.to_csv(args.output_csv, index=False)
print(f"Clustered dataset saved to {args.output_csv}")

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for c in range(n_clusters):
    plt.scatter(X_2d[cluster_labels == c, 0], X_2d[cluster_labels == c, 1], label=f"Cluster {c}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Clusters visualization")
plt.legend()
plt.tight_layout()
plt.savefig(args.output_plot, dpi=300)
print(f"Cluster plot saved to {args.output_plot}")
plt.close()
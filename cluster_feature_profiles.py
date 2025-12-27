import pandas as pd

#Load clustered dataset from Tasks 1–5
df = pd.read_csv("clustered_output.csv")

cluster_sizes = df['cluster'].value_counts().sort_index()
print("Cluster sizes:")
print(cluster_sizes)
summary = df.groupby('cluster').agg(['mean', 'median'])
print("\nCluster feature summary (mean and median of each feature):")
print(summary)

global_means = df.mean(numeric_only=True)
feature_profiles = {
    "feature_1": "shows higher overall activity",
    "feature_2": "indicates stronger intensity",
    "feature_3": "tends to be more interactive",
    "feature_4": "experiences frequent transitions",
    "feature_5": "demonstrates recurring patterns"
}

print("\n")
print("Feature Profiles:")
for c in sorted(df['cluster'].unique()):
    cluster_means = df[df['cluster'] == c].mean(numeric_only=True)
    diff = (cluster_means - global_means).abs().drop('cluster', errors='ignore')
    top_feature = diff.idxmax()
    tendency = "higher than average" if cluster_means[top_feature] > global_means[top_feature] else "lower than average"
    hint = feature_profiles.get(top_feature, top_feature)

    print(f"- Cluster {c} {hint}, which tends to be {tendency}, distinguishing this group from others.")
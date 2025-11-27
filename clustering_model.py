import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#loading preprocessed data
df = pd.read_csv("retail_customers_preprocessed.csv")

#feature selection
numeric_features = [
    "annual_income",
    "num_children",
    "num_teenagers",
    "days_since_last_purchase",
    "days_since_signup",
    "has_recent_complaint",
    "spend_wine", "spend_fruits", "spend_meat",
    "spend_fish", "spend_sweets", "spend_gold",
    "num_discount_purchases",
    "num_web_purchases", "num_catalog_purchases",
    "num_store_purchases", "web_visits_last_month"
]

X = df[numeric_features]

#data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#PCA (KEEP 90% VARIANCE)
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)

print("PCA components retained:", pca.n_components_)

#trying k values from 2 to 12
sil_scores = [] #silhouette score is how close/far a point is to other points in its cluster (ranges from -1 to 1)
inertias = [] #inertia is how far data points are from their assigned cluster center (sum of squared distances)
K_range = range(2, 13)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    sil_scores.append(sil)

    inertias.append(km.inertia_)
    print(f"K={k}, Silhouette={sil:.4f}")

#plotting sillouette vs k
plt.figure(figsize=(7, 4))
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores")
plt.grid(True)
plt.show()

#elbow plot
plt.figure(figsize=(7, 4))
plt.plot(K_range, inertias, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.grid(True)
plt.show()

#choosing best k automatically
best_k = K_range[np.argmax(sil_scores)]
print("Best k =", best_k)

#kmeans after choosing best k
kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
df["cluster_kmeans"] = kmeans.fit_predict(X_pca)

print("Final Silhouette Score (KMeans):",
      silhouette_score(X_pca, df["cluster_kmeans"]))

#agglomerative clustering
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
df["cluster_agg"] = agg.fit_predict(X_pca)

print("Final Silhouette Score (Agglomerative):",
      silhouette_score(X_pca, df["cluster_agg"]))

#summary
print("\n=== Cluster Summary (KMeans) ===")
print(df.groupby("cluster_kmeans")[numeric_features].mean())

print("\n=== Cluster Summary (Agglomerative) ===")
print(df.groupby("cluster_agg")[numeric_features].mean())

#PCA 2D plot
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=df["cluster_kmeans"], cmap="tab10", s=20)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters (PCA 2D)")
plt.grid(True)
plt.show()

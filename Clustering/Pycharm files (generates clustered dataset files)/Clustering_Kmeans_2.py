import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# 1. Load Dataset
df = pd.read_csv("retail_customers_preprocessed.csv")
print("Dataset loaded, shape:", df.shape)

# 2. Feature Selection
# Focusing on features that separate spending and purchase behavior
selected_features = [
    "annual_income",
    "spend_wine", "spend_fruits", "spend_meat", "spend_fish",
    "spend_sweets", "spend_gold",
    "num_web_purchases", "num_catalog_purchases", "num_store_purchases",
    "num_discount_purchases",
]

X = df[selected_features].copy()

# 3. Transform Skewed Features
# Most spend features are skewed; here we log-transform them
for col in ["spend_wine", "spend_fruits", "spend_meat", "spend_fish",
            "spend_sweets", "spend_gold"]:
    X[col] = np.log1p(X[col])  # log1p avoids log(0)

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. PCA (Keeping 90% variance)
pca = PCA(n_components=0.90, random_state=0)
X_pca = pca.fit_transform(X_scaled)
print("PCA components retained:", pca.n_components_)

# 6. KMeans Clustering (tuning k)
sil_scores = []
inertias = []
K_range = range(2, 13)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=0, n_init=20)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    sil_scores.append(sil)
    inertias.append(km.inertia_)
    print(f"K={k}, Silhouette={sil:.4f}")


# 7. Plot Silhouette Scores & Elbow
plt.figure(figsize=(7, 4))
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(K_range, inertias, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.grid(True)
plt.show()


# 8. Choose Best K Automatically

best_k = K_range[np.argmax(sil_scores)]
print("Best k =", best_k) #best k value here, according to silhouette score, is 2


# 9. Fit Final KMeans

kmeans_final = KMeans(n_clusters=best_k, random_state=0, n_init=20)
labels = kmeans_final.fit_predict(X_pca)

# Create a new dataset with clusters
df_clusters = df.copy()
df_clusters["cluster_kmeans"] = labels

# Save as NEW file
df_clusters.to_csv("retail_customers_with_2_clusters.csv", index=False)

final_sil = silhouette_score(X_pca, labels)
print("Final Silhouette Score (KMeans):", final_sil)

# 10. Cluster Summary
summary = df.groupby("cluster_kmeans")[selected_features].mean()
summary["count"] = df["cluster_kmeans"].value_counts()
print("\n=== Cluster Summary ===")
print(summary)

# 11. Visualization (2D PCA)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["cluster_kmeans"], cmap="tab10", s=25)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("KMeans Clusters (2D PCA)")
plt.grid(True)
plt.show()
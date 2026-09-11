import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Data
data = pd.read_csv("Iris.csv")

print("Dataset shape:", data.shape)
print(f"The total null values in data {data.isnull().sum().sum()} \n")
missing_values = data.isnull().sum()
total_cell = np.prod(data.shape)
print(
    f"The percentage of the missing values {(missing_values / total_cell) * 100} % \n"
)
print(data.describe())
print(data.info())
print(f" \n {data.head(5)} ")

# Drop 'Id' column if present
if "Id" in data.columns:
    data = data.drop("Id", axis=1)

# Dynamically identify species column and encode target labels
species_col = [col for col in data.columns if "species" in col.lower()][0]
label_encoder = LabelEncoder()
data["Species_Code"] = label_encoder.fit_transform(data[species_col])

print(data.describe())
print(f"data info : {data.info()}")

X = data.drop([species_col, "Species_Code"], axis=1)
y = data["Species_Code"]

# 2. Dimensionality Reduction with PCA
# Standardize features before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce 4D features (sepal/petal length & width) down to 2 principal components
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
print(
    f"\nExplained Variance Ratio by PC1 and PC2: {explained_variance.round(4)}"
)
print(f"Total Variance Retained: {sum(explained_variance) * 100:.2f}%\n")

# 3. Apply K-Means Clustering on PCA Features (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

print("\nFirst 10 predicted clusters:")
print(clusters[:10])

print("\nFirst 10 actual species codes:")
print(y.iloc[:10].values)

ari = adjusted_rand_score(y, clusters)
print("\nAdjusted Rand Index (ARI):", ari)

crosstab = pd.crosstab(
    data[species_col],
    clusters,
    rownames=["True Species"],
    colnames=["Predicted Cluster"],
)
print("\nContingency Matrix:")
print(crosstab)

# 4. Visualize PCA Clusters vs Actual Species
plt.figure(figsize=(14, 5))

# Subplot 1: Predicted K-Means Clusters in 2D PCA Space
plt.subplot(1, 2, 1)
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    alpha=0.6,
    edgecolor="k",
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroids",
)
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("K-Means Clusters on PCA Space (k=3)")
plt.legend()

# Subplot 2: Actual Species Labels in 2D PCA Space
plt.subplot(1, 2, 2)
scatter_true = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    alpha=0.6,
    edgecolor="k",
)
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("Actual Iris Species Labels (PCA Space)")

# Add species legend mapping
handles, _ = scatter_true.legend_elements()
target_names = label_encoder.classes_
plt.legend(handles, target_names, title="Species")

plt.tight_layout()
plt.show()
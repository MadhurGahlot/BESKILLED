# Mini Project 3: Iris Flower Clustering Project
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("D:/BESKILLED/week3/Iris.csv")

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

# Dynamically identify species column
species_col = [col for col in data.columns if "species" in col.lower()][0]

# Label encode target species string
label_encoder = LabelEncoder()
data["Species_Code"] = label_encoder.fit_transform(data[species_col])

print(data.describe())
print(f"data info : {data.info()}")

X = data.drop([species_col, "Species_Code"], axis=1)
y = data["Species_Code"]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

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

# Locate petal length and width columns for visualization
petal_length_col = [
    c for c in X.columns if "petal" in c.lower() and "length" in c.lower()
][0]
petal_width_col = [
    c for c in X.columns if "petal" in c.lower() and "width" in c.lower()
][0]

petal_length_idx = X.columns.get_loc(petal_length_col)
petal_width_idx = X.columns.get_loc(petal_width_col)

plt.figure(figsize=(14, 5))

# Subplot 1: Predicted Clusters
plt.subplot(1, 2, 1)
plt.scatter(
    X[petal_length_col],
    X[petal_width_col],
    c=clusters,
    cmap="viridis",
    alpha=0.6,
    edgecolor="k",
)
plt.scatter(
    kmeans.cluster_centers_[:, petal_length_idx],
    kmeans.cluster_centers_[:, petal_width_idx],
    c="red",
    marker="X",
    s=200,
    label="Centroids",
)
plt.xlabel(petal_length_col)
plt.ylabel(petal_width_col)
plt.title("K-Means Predicted Clusters (k=3)")
plt.legend()

# Subplot 2: Actual Species
plt.subplot(1, 2, 2)
scatter_true = plt.scatter(
    X[petal_length_col],
    X[petal_width_col],
    c=y,
    cmap="viridis",
    alpha=0.6,
    edgecolor="k",
)
plt.xlabel(petal_length_col)
plt.ylabel(petal_width_col)
plt.title("Actual Iris Species Labels")

# Add species legend mapping
handles, _ = scatter_true.legend_elements()
target_names = label_encoder.classes_
plt.legend(handles, target_names, title="Species")

plt.tight_layout()
plt.show()
## TASK - 8
---
Objective: Perform unsupervised learning with K-Means clustering.
Tools: Scikit-learn, Pandas, Matplotlib 
step as follows 
1. 📂 Loading & preprocessing the data
2. 📊 Clustering using **K-Means** on `Age` and `Spending Score`
3. 📉 Visualizing **K-Means** clusters
4. 📈 3D clustering using **PCA**
5. 🧩 Clustering using **DBSCAN**
6. 📏 Evaluating using **Silhouette Score**

---

## 🪜 STEP-BY-STEP EXPLANATION

---

### ✅ **STEP 1: Load and Preprocess the Dataset**

```python
df = pd.read_csv('/mnt/data/Mall_Customers.csv')
```

* Load the CSV using Pandas.
* Drop `CustomerID` — it's just an identifier, not useful for clustering.
* Convert `Genre` (Male/Female) to numeric (0/1) if present.

```python
df_clean = df.drop(columns=['CustomerID'])
df_clean['Genre'] = df_clean['Genre'].map({'Male': 0, 'Female': 1})
```

---

### ✅ **STEP 2: Select Two Features (`Age` and `Spending Score`)**

```python
X_selected = df_clean[['Age', 'Spending Score (1-100)']]
```

* We use just two features to keep things simple and visual.
* These two features show interesting clustering patterns.

---

### ✅ **STEP 3: Standardize the Data**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
```

* Clustering algorithms like K-Means & DBSCAN are sensitive to **scale**.
* Standardization (mean = 0, std = 1) ensures all features contribute equally.

---

### ✅ **STEP 4: Apply K-Means Clustering**

```python
kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(X_scaled)
```

* `n_clusters=3` means we ask K-Means to find 3 clusters.
* Each point is assigned to one of those clusters.

---

### ✅ **STEP 5: Visualize K-Means Clusters (2D Plot)**

```python
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=kmeans_labels)
```

* A 2D scatterplot of clusters.
* Each color = one cluster found by K-Means.

---

### ✅ **STEP 6: Apply PCA to All Features for 3D Clustering View**

```python
X_all_scaled = StandardScaler().fit_transform(df_clean)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_all_scaled)
```

* We reduce high-dimensional data (like 5–6 columns) to **3D** using **PCA**.
* PCA keeps the **maximum variance** for plotting and viewing.

---

### ✅ **STEP 7: K-Means on Full Dataset + 3D Plot**

```python
kmeans_3d = KMeans(n_clusters=5)
labels_3d = kmeans_3d.fit_predict(X_all_scaled)

# 3D scatter plot
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=labels_3d)
```

* Clustering is done in full dimensional space.
* We **visualize in 3D** using PCA-reduced dimensions.

---

### ✅ **STEP 8: Apply DBSCAN for Density-Based Clustering**

```python
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)
```

* DBSCAN finds clusters based on **density**.
* No need to specify number of clusters.
* Points that don't belong to any cluster are labeled **-1 (noise)**.

---

### ✅ **STEP 9: Visualize DBSCAN Clusters (2D Plot)**

```python
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=db_labels)
```

* Similar to K-Means plot, but DBSCAN may show **outliers (noise)**.
* More flexible than K-Means — handles irregular shapes and noise.

---

### ✅ **STEP 10: Evaluate Clustering Quality Using Silhouette Score**

```python
silhouette_score(X_scaled, kmeans_labels)  # for K-Means
silhouette_score(X_scaled[db_labels != -1], db_labels[db_labels != -1])  # for DBSCAN
```

* Silhouette score ranges from **-1 to +1**:

  * **+1** → clearly separated clusters
  * **0** → overlapping clusters
  * **-1** → incorrect assignment
* For DBSCAN, we **exclude noise points (-1)** from score calculation.

---

## ✅ Summary of Techniques Used

| Method               | What It Does                             | Strength                    |
| -------------------- | ---------------------------------------- | --------------------------- |
| **K-Means**          | Assigns points to fixed K clusters       | Fast and simple             |
| **PCA**              | Reduces data to 2D or 3D for plotting    | Helps visualization         |
| **DBSCAN**           | Finds clusters by density, detects noise | Works with irregular shapes |
| **Silhouette Score** | Evaluates clustering quality             | No labels needed            |

---

## TASK -7
---

## 🧠 Objective

Use SVM to:

* Perform both **linear** and **non-linear (RBF)** classification
* Visualize decision boundaries
* Tune hyperparameters like `C` and `gamma`
* Evaluate model using cross-validation

---

### ✅ **Step 1: Load and Prepare the Dataset**

```python
df = pd.read_csv('/mnt/data/breast-cancer.csv')
```

* Read the dataset into a DataFrame.
* Inspect column names (`df.columns`) and head (`df.head()`).
* Determine which column is the **target** (`diagnosis`, etc.).

---

### ✅ **Step 2: Encode and Scale Data**

```python
# Encode categorical target
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

# Split features and labels
X = df.drop(columns=[target_col])
y = df[target_col]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

* Many models perform better when data is **standardized** (mean 0, std 1).
* LabelEncoder converts text labels like 'M', 'B' to 0 and 1.

---

### ✅ **Step 3: Reduce to 2D Using PCA**

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

* PCA reduces dimensionality for **visualization**.
* We only use 2 principal components for plotting.

---

### ✅ **Step 4: Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
```

* Split 80% training and 20% test data.

---

### ✅ **Step 5: Train SVM Models**

```python
# Linear kernel SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# RBF (non-linear) kernel SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)
```

* `C` controls margin hardness (regularization).
* `gamma` controls curve of decision boundary (used in RBF).

---

### ✅ **Step 6: Visualize Decision Boundaries**

```python
def plot_decision_boundary(model, X, y, title):
    ...
plot_decision_boundary(svm_linear, X_pca, y, "SVM with Linear Kernel")
plot_decision_boundary(svm_rbf, X_pca, y, "SVM with RBF Kernel")
```

* This function creates a mesh grid and uses the trained SVM to **predict** for every point.
* Then, it plots the decision boundary and your original data points.

---

### ✅ **Step 7: Tune Hyperparameters with GridSearchCV**

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001]
}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train, y_train)
```

* Grid search tests combinations of `C` and `gamma` to find the best.
* `cv=5` does 5-fold cross-validation.

---

### ✅ **Step 8: Evaluate the Best Model**

```python
svm_best = grid.best_estimator_
y_pred = svm_best.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

* Reports accuracy, precision, recall, and F1-score.
* Confusion matrix shows how many true/false positives/negatives.

---

### ✅ **Step 9: Cross-Validation Accuracy**

```python
cv_scores = cross_val_score(svm_best, X_pca, y, cv=5)
print("Cross-Validation Accuracy (mean):", cv_scores.mean())
```

* Validates how the model performs across multiple folds of data.
* Ensures the model generalizes well.

---

### ✅ **Step 10: Visualize Evaluation**

```python
# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True)

# Classification Report as Heatmap
sns.heatmap(report_df.iloc[:-1, :-1], annot=True)
```

* Heatmaps are intuitive to read.
* You can see how balanced and accurate your predictions are.

---

## ✅ Summary of What You Achieve

| Task                                 | Done |
| ------------------------------------ | ---- |
| Load and preprocess dataset          | ✅    |
| Train SVM with linear and RBF kernel | ✅    |
| Visualize decision boundaries        | ✅    |
| Hyperparameter tuning                | ✅    |
| Evaluate with classification metrics | ✅    |
| Cross-validation                     | ✅    |

---

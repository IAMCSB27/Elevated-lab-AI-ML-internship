## TASK 6
---

## 🔢 Step 1: Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
```

These libraries help with:

* **Data loading and preprocessing** (`pandas`, `numpy`)
* **Plotting** (`matplotlib`, `seaborn`)
* **Machine learning tasks** (`scikit-learn`)

---

## 📂 Step 2: Load the Dataset

```python
df = pd.read_csv("Iris.csv")
df.drop(columns=['Id'], inplace=True)
```

* Load the dataset into a DataFrame.
* Drop the `Id` column since it's not useful for classification.

---

## 📊 Step 3: Split Features and Labels

```python
X = df.drop(columns=['Species'])  # Features
y = df['Species']                # Labels
```

* `X` contains the input features like SepalLength, PetalWidth, etc.
* `y` contains the class labels: `Iris-setosa`, `Iris-versicolor`, and `Iris-virginica`.

---

## 📏 Step 4: Normalize Features

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

* Standardize the feature values to **mean = 0** and **std = 1** to ensure fair distance calculations in KNN.

---

## 🔀 Step 5: Split Data into Train/Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
```

* Split the data into 80% for training and 20% for testing.

---

## 🔍 Step 6: Train and Evaluate KNN for Different K

```python
k_range = range(1, 21)
accuracies = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
```

* Try different values of `k` (number of neighbors).
* Train the model, make predictions, and calculate accuracy for each k.

---

## 📈 Step 7: Plot Accuracy vs. K

```python
plt.plot(k_range, accuracies, marker='o')
```

* Visualize how model performance changes with different values of K.
* Helps you **choose the best K**.

---

## ✅ Step 8: Train Final Model with Best K

```python
best_k = k_range[np.argmax(accuracies)]
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

* Use the `best_k` that gave the highest accuracy.
* Train and predict using this K.

---

## 📉 Step 9: Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

* Shows how many predictions were correct vs. incorrect for each class.
* Useful for spotting specific misclassifications.

---

## 🌐 Step 10: Visualize Decision Boundaries (with PCA)

```python
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
```

* PCA reduces the original 4D feature space to 2D for visualization.

```python
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42)

knn_pca = KNeighborsClassifier(n_neighbors=best_k)
knn_pca.fit(X_train_pca, y_train_pca)
```

* Re-train KNN on the reduced 2D features.

```python
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
```

* Create a meshgrid to plot the decision surface.

```python
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y)
```

* Plot decision regions and data points to show how KNN classifies in 2D space.

---

## 📌 Summary

| Step | Description                             |
| ---- | --------------------------------------- |
| 1    | Import libraries                        |
| 2    | Load Iris dataset                       |
| 3    | Split features/labels                   |
| 4    | Normalize features                      |
| 5    | Train/test split                        |
| 6    | Train KNN for multiple K                |
| 7    | Plot accuracy vs K                      |
| 8    | Train final model with best K           |
| 9    | Evaluate using confusion matrix         |
| 10   | Visualize decision boundaries using PCA |

---

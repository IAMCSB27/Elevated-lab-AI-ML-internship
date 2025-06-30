## Task-5
---

## 🎯 **Objective**

Learn to build and evaluate **tree-based models** for **classification** using:

* `DecisionTreeClassifier`
* `RandomForestClassifier`
* Visualization tools (Matplotlib, Graphviz-style tree)
* Model evaluation and cross-validation

---

## 🧩 Step-by-Step Explanation

---

### 🔹 Step 1: Import Libraries & Load Dataset

```python
import pandas as pd
df = pd.read_csv("heart.csv")
```

* Loads the **Heart Disease dataset**
* Target column: `target` (1 = heart disease, 0 = no disease)
* All columns are numeric → no encoding required.

---

### 🔹 Step 2: Define Features and Target

```python
X = df.drop('target', axis=1)
y = df['target']
```

* `X` contains all features (like age, cp, thalach, oldpeak, etc.)
* `y` is the target (classification: disease or not)

---

### 🔹 Step 3: Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(..., test_size=0.2)
```

* Splits data into 80% training and 20% testing
* Ensures we evaluate the model on unseen data

---

### 🔹 Step 4: Train Decision Tree Classifier (Default)

```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
```

* Builds a **Decision Tree** model
* Default: no depth restriction → can **overfit** on small datasets

---

### 🔹 Step 5: Visualize the Tree (First 3 Levels)

```python
from sklearn.tree import plot_tree
plot_tree(dt, filled=True, max_depth=3)
```

* Visualizes the first few levels of the decision process
* Helps understand **which features** the tree splits on early

---

### 🔹 Step 6: Evaluate the Decision Tree

```python
from sklearn.metrics import accuracy_score, classification_report
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
```

* Measures how accurately the tree predicts on test data
* Reports **precision**, **recall**, **f1-score** for each class

---

### 🔹 Step 7: Prune the Tree to Control Overfitting

```python
dt_pruned = DecisionTreeClassifier(max_depth=4)
```

* Restricts the tree's depth → fewer splits → **less overfitting**
* May reduce accuracy slightly, but **improves generalization**

---

### 🔹 Step 8: Train a Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
```

* Builds an **ensemble of decision trees**
* Each tree sees a random subset of features & data
* Final prediction = **majority vote** of all trees
* More robust and accurate than single decision trees

---

### 🔹 Step 9: Feature Importance from Random Forest

```python
rf.feature_importances_
```

* Random Forest calculates **how important** each feature is in making decisions
* Importance = how much a feature reduces impurity across all trees

#### Visualization:

```python
sns.barplot(x='Importance', y='Feature', data=feature_df.head(10))
```

* Bar plot showing top predictors of heart disease like:

  * `cp` (chest pain type)
  * `thalach` (max heart rate)
  * `ca` (number of vessels colored by fluoroscopy)

---

### 🔹 Step 10: Cross-Validation for Robust Evaluation

```python
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5)
```

* Performs **5-fold cross-validation**:

  * Splits data into 5 parts, trains on 4, tests on 1 — repeats 5 times
* Returns more **stable** accuracy estimate than single test set

---

## ✅ Results Summary

| Model                     | Accuracy (Test Set) | CV Accuracy (5-fold) |
| ------------------------- | ------------------- | -------------------- |
| Decision Tree (default)   | \~98.5% *(overfit)* | lower (unstable)     |
| Pruned Decision Tree      | \~80.0%             | 83.4%                |
| Random Forest (100 trees) | \~98.5% ✅           | **99.7%** ✅          |

---

### 🎯 Final Takeaways:

* **Decision Trees** are easy to interpret but prone to **overfitting**
* **Pruning (limiting depth)** helps make trees generalize better
* **Random Forests** are more powerful and robust
* **Feature Importance** helps explain predictions
* **Cross-validation** ensures your model performs well consistently

---


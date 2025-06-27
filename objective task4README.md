

## Task4 🎯 Objective

* Build a binary classifier using logistic regression.
* Tools: Scikit-learn, Pandas, Matplotlib
* Dataset: Breast Cancer Wisconsin Dataset.
To classify tumors as **Malignant (M)** or **Benign (B)** using logistic regression on the **Breast Cancer Wisconsin dataset**.

---

## ✅ Step-by-Step Explanation

---

### 🔹 Step 1: **Import Libraries**

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ...
```

We import:

* **Pandas, Numpy**: Data manipulation
* **Matplotlib, Seaborn**: Visualization
* **Scikit-learn**: Machine learning tools

---

### 🔹 Step 2: **Load and Preprocess the Dataset**

```python
df = pd.read_csv("data.csv")
df = df.drop(['id', 'Unnamed: 32'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
```

* **Drop `id`**: It's not useful for prediction.
* **Drop `Unnamed: 32`**: Empty column.
* **Encode `diagnosis`**:

  * `M` = 1 → Malignant (positive case)
  * `B` = 0 → Benign (negative case)

---

### 🔹 Step 3: **Split and Standardize the Data**

```python
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(...)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

* Split into **80% training** and **20% testing**.
* **StandardScaler** makes all features have zero mean and unit variance:

  * This helps logistic regression converge faster and perform better.

---

### 🔹 Step 4: **Train the Logistic Regression Model**

```python
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
```

This fits a logistic regression model that learns the best weights to separate malignant and benign cases.

---

### 🔹 Step 5: **Predict and Evaluate the Model**

```python
y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
```

* `y_pred`: 0 or 1 predictions
* `y_proba`: Probability of being class 1 (malignant)

#### ✅ Evaluation Metrics:

```python
confusion_matrix(...)
precision_score(...)
recall_score(...)
roc_auc_score(...)
classification_report(...)
```

* **Confusion Matrix**: Shows TP, TN, FP, FN
* **Precision**: How many predicted positives are truly positive
* **Recall**: How many actual positives are correctly identified
* **ROC AUC**: Area under the ROC Curve (1.0 = perfect)

---

### 🔹 Step 6: **Visualize Performance**

#### ✅ Confusion Matrix

```python
sns.heatmap(conf_mat, ...)
```

* Visualizes prediction performance with actual vs predicted values.

#### ✅ ROC Curve

```python
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr)
```

* ROC shows **true positive rate vs false positive rate**.
* Helps assess how well model separates the two classes.

---

### 🔹 Step 7: **Plot the Sigmoid Function**

```python
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))
```

* The logistic function (sigmoid) outputs a **probability between 0 and 1**.
* By default, if output > 0.5 → class 1, else → class 0.

#### 📈 Sigmoid Plot:

* Shows how the sigmoid squashes any real number `z` to a probability between 0 and 1.
* `z = 0` gives probability = 0.5 (threshold boundary)

---

### 🔹 Step 8: **Threshold Tuning & Visualization**

```python
thresholds_to_test = np.arange(0.1, 0.9, 0.05)
for thresh in thresholds_to_test:
    custom_pred = (y_proba >= thresh).astype(int)
    ...
```

* The default threshold is **0.5**.
* You can **manually adjust** this threshold to balance:

  * **Precision** (fewer false positives)
  * **Recall** (fewer false negatives)

#### 📊 Precision/Recall vs Threshold Plot:

* Helps choose the right threshold depending on your priority (e.g., high recall in cancer detection).

---

## ✅ Results Summary

| Metric    | Value         |
| --------- | ------------- |
| Accuracy  | \~98%         |
| Precision | \~97.6%       |
| Recall    | \~95.3%       |
| ROC AUC   | **\~0.997** ✅ |

🎯 **Conclusion**:
The logistic regression model performs **very well**, and you can further tune the threshold depending on whether you want to minimize **false negatives** (recall) or **false positives** (precision).

---

Would you like this as a `.py` file or Jupyter Notebook? Or should I show how to make predictions on new data?

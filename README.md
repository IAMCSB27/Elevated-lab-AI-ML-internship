# Task-1 -🧹 Titanic Dataset – Data Cleaning & Preprocessing

## 🎯 Objective
Prepare the raw Titanic dataset for machine learning by cleaning and preprocessing:
- Handle missing values
- Encode categorical features
- Standardize numerical features
- Detect and remove outliers

---

## 🛠️ Tools Used
- **Language:** Python
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

---

## 📂 Dataset
- **Source:** Titanic dataset (`Titanic-Dataset.csv`)
- **Total Records:** 891
- **Features Include:**  
  - `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

---

## 🔧 Steps Performed

### 1. Data Exploration
- Used `df.info()` and `df.isnull().sum()` to examine structure and missing values

### 2. Handling Missing Values
- Filled missing `Age` values with the median
- Dropped `Cabin` due to ~77% missing values
- Filled missing `Embarked` values with the mode

### 3. Encoding Categorical Features
- `Sex` encoded using `LabelEncoder` (male = 1, female = 0)
- `Embarked` encoded using one-hot encoding (`drop_first=True`)

### 4. Feature Scaling
- Standardized `Age` and `Fare` columns using `StandardScaler` (Z-score scaling)

### 5. Outlier Detection & Removal
- Used Seaborn boxplots to detect outliers
- Removed outliers from `Age` and `Fare` using the IQR method

### 6. Final Output
- Dataset cleaned, transformed, and ready for machine learning model training

---

## 📊 Sample Visualizations
- **Boxplots** of `Age` and `Fare` were used to visualize outliers

---

## ✅ Final Result
- ✅ No missing values  
- ✅ Categorical features encoded  
- ✅ Numerical features standardized  
- ✅ Outliers removed  
- ✅ Dataset ready for ML models

---

## Task-3 🎯 **Objective**

To understand and implement:

* **Simple Linear Regression** (one feature)
* **Multiple Linear Regression** (multiple features)
  Using:
* Tools: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

## 🧩 Step-by-Step Process

---

### 🔹 Step 1: **Import Libraries and Load the Dataset**

```python
import pandas as pd
df = pd.read_csv("Housing.csv")
```

This loads the housing dataset which includes house price, area, number of bedrooms, etc.

---

### 🔹 Step 2: **Preprocess the Data**

* The dataset includes **categorical features** like `mainroad`, `guestroom`, etc.
* These need to be converted into numerical form using **One-Hot Encoding**:

```python
df_encoded = pd.get_dummies(df, drop_first=True)
```

`drop_first=True` avoids the dummy variable trap by removing one column per category.

---

## 🟡 Part A: Simple Linear Regression

---

### 🎯 Goal:

Predict `price` based on just **one feature** — `area`.

---

### 🔸 Step A1: **Split Data**

```python
X_simple = df_encoded[['area']]
y = df_encoded['price']

from sklearn.model_selection import train_test_split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
```

---

### 🔸 Step A2: **Train the Model**

```python
from sklearn.linear_model import LinearRegression
lr_simple = LinearRegression()
lr_simple.fit(X_train_s, y_train_s)
```

---

### 🔸 Step A3: **Predict and Evaluate**

```python
y_pred_s = lr_simple.predict(X_test_s)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)
```

* **MAE** = Average of absolute errors
* **MSE** = Average of squared errors
* **R²** = Model accuracy (how well the regression explains data)

---

### 🔸 Step A4: **Visualize the Regression Line**

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_test_s['area'], y=y_test_s, label='Actual')
plt.plot(X_test_s['area'], y_pred_s, color='red', label='Regression Line')
plt.title('Simple Linear Regression: Price vs Area')
plt.legend()
plt.show()
```

---

### 📊 Simple Model Results (Example):

* MAE ≈ ₹14.7 lakhs
* R² ≈ 0.27 (Low accuracy)
* Coefficient ≈ ₹426/sq.ft
* Intercept ≈ ₹25.1 lakhs

---

## 🟢 Part B: Multiple Linear Regression

---

### 🎯 Goal:

Predict `price` using **all available features**.

---

### 🔸 Step B1: **Split Data**

```python
X_multi = df_encoded.drop('price', axis=1)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
```

---

### 🔸 Step B2: **Train the Model**

```python
lr_multi = LinearRegression()
lr_multi.fit(X_train_m, y_train_m)
```

---

### 🔸 Step B3: **Predict and Evaluate**

```python
y_pred_m = lr_multi.predict(X_test_m)

mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)
```

---

### 🔸 Step B4: **Interpret Coefficients**

```python
coeff_df = pd.DataFrame({
    'Feature': X_multi.columns,
    'Coefficient': lr_multi.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)
```

This shows the top features influencing house price the most.

---

### 📊 Multiple Model Results (Example):

* MAE ≈ ₹9.7 lakhs
* R² ≈ **0.65** (65% variance explained — good!)
* Top influential features:

  * Bathrooms
  * Airconditioning
  * Hot Water Heating
  * Preferred Area
  * Basement, Furnishing Status, etc.

---

## ✅ Summary:

| Model Type          | MAE         | R² Score | Key Insight                            |
| ------------------- | ----------- | -------- | -------------------------------------- |
| Simple Regression   | ₹14.7 Lakhs | 0.27     | Only *area* explains 27% of price      |
| Multiple Regression | ₹9.7 Lakhs  | 0.65     | Multiple features explain 65% of price |

---

## task 4 🎯 Objective
Build a binary classifier using logistic regression.
Tools: Scikit-learn, Pandas, Matplotlib
Dataset: Breast Cancer Wisconsin Dataset.
To classify tumors as **Malignant (M)** or **Benign (B)** using logistic regression on the **Breast Cancer Wisconsin dataset**.

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

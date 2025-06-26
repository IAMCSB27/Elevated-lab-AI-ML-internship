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

## Task-2  🔍 EDA on Titanic Dataset

---

### **🎯 Objective**

To understand the data better by summarizing it statistically and visually. This helps in detecting patterns, trends, and anomalies, and guides feature engineering and modeling decisions.

---

### ✅ **Step 1: Load the Dataset**

```python
df = pd.read_csv("Titanic-Dataset.csv")
```

* Loads the dataset into a DataFrame using `pandas`.

---

### ✅ **Step 2: Summary Statistics**

```python
df.describe(include='all')
df.isnull().sum()
```

* `describe()` gives summary stats:

  * Mean, median, standard deviation (for numerical)
  * Unique values, top category, frequency (for categorical)
* `isnull().sum()` identifies missing values in each column.

---

### ✅ **Step 3: Visualize Distributions (Histograms & Boxplots)**

#### 🔹 **Histograms**

```python
df[numeric_cols].hist(bins=20)
```

* Helps visualize the distribution of numerical features like:

  * `Age`, `Fare`, `SibSp` (siblings/spouses), `Parch` (parents/children)

#### 🔹 **Boxplots**

```python
sns.boxplot(y=df['Age'])
```

* Highlights **outliers** and distribution shape for each numeric column.

---

### ✅ **Step 4: Feature Relationships (Correlation Matrix & Pairplot)**

#### 🔹 **Correlation Matrix**

```python
df.corr()
sns.heatmap(corr_matrix)
```

* Shows how strongly numerical features are correlated.

  * For example, `Fare` and `Pclass` may be inversely correlated.

#### 🔹 **Pairplot**

```python
sns.pairplot(df[['Age', 'Fare', 'Survived']], hue='Survived')
```

* Plots scatterplots for feature pairs.
* Colored by survival status to find clustering patterns.

---

### ✅ **Step 5: Trend/Pattern Detection (Barplots)**

#### 🔹 **Survival by Class**

```python
sns.barplot(x='Pclass', y='Survived', data=df)
```

* Shows how survival rate changes across `Pclass` (1st, 2nd, 3rd).

#### 🔹 **Survival by Sex**

```python
sns.barplot(x='Sex', y='Survived', data=df)
```

* Reveals gender impact — typically, women had higher survival rates.

---

### ✅ **Step 6: Optional – Interactive Plot (Plotly)**

```python
px.scatter(df, x='Age', y='Fare', color='Survived')
```

* Provides a hoverable, zoomable scatter plot.
* Great for dashboards or interactive Jupyter notebooks.

---

## 🔚 Final Inference from EDA

* **Age**, **Fare**, **Sex**, and **Pclass** are meaningful predictors for survival.
* Outliers and missing values are present and need cleaning (already done in preprocessing).
* Women and upper-class passengers had higher survival chances.
* Most features are not strongly correlated — good for model independence.


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


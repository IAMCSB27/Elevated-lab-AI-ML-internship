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

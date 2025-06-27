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



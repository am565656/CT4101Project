# imports
import pandas as pd
import numpy as np

# load the dataset
df = pd.read_csv("data/Maternal Health Risk Data Set.csv")

# dataset shape
print("Dataset shape:", df.shape)

# variable types
print("Variable types:\n", df.dtypes)

# missing values
print("Missing values:\n", df.isnull().sum())

# duplicates
print("Duplicate rows:", df.duplicated().sum())
# drop duplicates
df = df.drop_duplicates()
# checking the shape of output after dropping duplicates
print(df["RiskLevel"].value_counts(normalize=True))

# outlier detection
numeric_cols = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    print(f"{col}: {outliers}")

# removing outliers
df = df[df["HeartRate"] >= 30]

# target encoding
df["RiskLevel"] = df["RiskLevel"].map({"low risk": 0, "mid risk": 1, "high risk": 2})

# feature-target split
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# dummy baseline model: stratified classifier
from sklearn.dummy import DummyClassifier
baseline_model = DummyClassifier(strategy="stratified", random_state=42)

# shallow decision tree model
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)

# dummy results
from sklearn.model_selection import cross_val_score
baseline_scores = cross_val_score(baseline_model, X_scaled, y, cv=5, scoring="accuracy")
print("Baseline model accuracy:", baseline_scores.mean())

# decision tree results
tree_scores = cross_val_score(tree_model, X_scaled, y, cv=5, scoring="accuracy")
print("Decision tree accuracy:", tree_scores.mean())



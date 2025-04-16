# Sanika Prashant Deshmukh, date - 04-11-25, Assignment 1, Testing unseen data on RF3 model with KNN imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import numpy as np
# ----------------- Load Data --------------------
data = pd.read_csv("iac-atmospheres.csv")
blind_data = pd.read_csv("iac-atmospheres-blind.csv")

X_full = data.drop(columns=["name", "type", "water"]) #dropping name, type and water columns
y_full = data["water"] #target output
X_clean = X_full[~X_full.isna().any(axis=1)] #only the rows from X_full that do not have missing values
y_clean = y_full[~X_full.isna().any(axis=1)] #selects only those rows in y_full where the row in X_full that do not have missing value

# Prepare blind data
X_blind = blind_data.drop(columns=["name", "type"]) #dropping name and type columns
X_blind = X_blind.reset_index(drop=True) #reset index to avoid misalignment


# ----------------- Missing Value Strategy --------------------
knn_imputer = KNNImputer()
knn_imputer.fit(X_clean) #fit KNN imputer on clean data
X_blind_imputed = pd.DataFrame(knn_imputer.transform(X_blind), columns=X_blind.columns) #transform blind data using the imputer fitted on clean data


rf3 = RandomForestClassifier(n_estimators=200, max_depth=5, max_features='sqrt', min_samples_leaf=2, random_state=42)
rf3.fit(X_clean, y_clean) #fit the model on clean data

# Predict on blind data
blind_preds = rf3.predict(X_blind_imputed) #predict on blind data

# Combine predictions with planet names
output_df = blind_data[["name"]].copy() #copy the name column from blind data
output_df["predicted_water"] = blind_preds #results of prediction

# Show result and optionally save
print("\n📊 Final predictions on blind set:")
print(output_df.head())


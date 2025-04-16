# Sanika Prashant Deshmukh, Assignment 1 - Extra Credit Task
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import numpy as np

# --------------- Load Dataset ---------------
data = pd.read_csv("iac-atmospheres.csv")
X = data.drop(columns=["name", "type", "water"])
y = data["water"]

# Identify missing and clean rows
missing_mask = X.isna().any(axis=1)
X_missing = X[missing_mask].reset_index(drop=True)
y_missing = y[missing_mask].reset_index(drop=True)
X_clean = X[~missing_mask].reset_index(drop=True)
y_clean = y[~missing_mask].reset_index(drop=True)

# Combine everything for random sampling of 200 training rows
X_full = X.reset_index(drop=True)
y_full = y.reset_index(drop=True)

# --------------- Helper Function for Imputation-Based Methods ---------------
def impute_train_and_evaluate(imputer, X_train, y_train, X_test, y_test, test_missing_mask):
    imputer.fit(X_train)
    X_train_filled = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_test_filled = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        max_features='sqrt',
        min_samples_leaf=2,
        random_state=0
    )
    model.fit(X_train_filled, y_train)
    preds = model.predict(X_test_filled)
    
    return (
        accuracy_score(y_test, preds),
        accuracy_score(y_test[test_missing_mask], preds[test_missing_mask])
    )

# --------------- Main Loop Over Seeds ---------------
results_extra = defaultdict(list)

for seed in range(10):
    print(f"\nSeed {seed} - Training on full data (with missing rows allowed)")

    # Select 200 random training samples from the full data (can include missing)
    X_train, _, y_train, _ = train_test_split(X_full, y_full, train_size=200, random_state=seed)

    # Use 12 clean + all 44 missing as test set
    X_test_clean = X_clean.sample(n=12, random_state=seed)
    y_test_clean = y_clean.loc[X_test_clean.index]
    X_test = pd.concat([X_test_clean, X_missing]).reset_index(drop=True)
    y_test = pd.concat([y_test_clean, y_missing]).reset_index(drop=True)
    test_missing_mask = X_test.isna().any(axis=1)

    # --- Method C: Drop rows with missing in training, predict only on clean test ---
    X_train_c = X_train.dropna()
    y_train_c = y_train.loc[X_train_c.index]
    
    X_test_c = X_test[~test_missing_mask]
    y_test_c = y_test[~test_missing_mask]
    
    model_c = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        max_features='sqrt',
        min_samples_leaf=2,
        random_state=0
    )
    model_c.fit(X_train_c, y_train_c)
    y_pred_c = model_c.predict(X_test_c)
    
    results_extra["Drop Missing Rows"].append((accuracy_score(y_test_c, y_pred_c), 0))  # Missing = 0 because not predicted

    # --- Method D: Mean Imputation ---
    results_extra["Mean Imputation"].append(
        impute_train_and_evaluate(SimpleImputer(strategy='mean'), X_train, y_train, X_test, y_test, test_missing_mask)
    )

    # --- Method E: Iterative Imputation ---
    results_extra["Multiple Imputation"].append(
        impute_train_and_evaluate(IterativeImputer(max_iter=5, random_state=0), X_train, y_train, X_test, y_test, test_missing_mask)
    )

    # --- Method F: KNN Imputation ---
    results_extra["KNN Imputation"].append(
        impute_train_and_evaluate(KNNImputer(), X_train, y_train, X_test, y_test, test_missing_mask)
    )

# --------------- Display Final Results ---------------
summary_df = pd.DataFrame({
    method: [f"{np.mean([r[0] for r in scores]):.4f} (Full)", f"{np.mean([r[1] for r in scores]):.4f} (Missing)"]
    for method, scores in results_extra.items()
}).T.rename(columns={0: "Avg Accuracy (Full Test)", 1: "Avg Accuracy (Missing Rows Only)"})

print("\n✅ Final Evaluation Results (Training on Data with Missing Values):")
print(summary_df.to_string())

# Sanika Prashant Deshmukh, date - 04-11-25, Assignment 1, Training and handlig missing data for test set and looping through 10 splits 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pandas as pd
import numpy as np

# ----------------- Load Data --------------------

#dataset is in the same directory as this script

data = pd.read_csv("iac-atmospheres.csv") 
X_total = data.drop(columns=['name', 'type', 'water']) #dropping name, type and water columns
y_total = data['water'] #target output

rows_with_missing = X_total.isna().sum(axis=1) > 0#rows with missing values
# Creating mask for rows with missing values
X_missing_rows = X_total[rows_with_missing].reset_index(drop=True) #only the rows from X_total that have missing values
y_missing_rows = y_total[rows_with_missing].reset_index(drop=True) #selects only those rows in y_total where the row in X_total has a missing value

X_clean_rows = X_total[~rows_with_missing].reset_index(drop=True) #only the rows from X_total that do not have missing values
y_clean_rows = y_total[~rows_with_missing].reset_index(drop=True) #selects only those rows in y_total where the row in X_total that do not have missing value

# ----------------- Missing Value Strategies --------------------

def majority(model, test_data, test_labels, train_labels, missing_rows_mask): #MAJORITY - most common class
    
    most_common = train_labels.mode()[0] #gets the most frequent class in training set
    test_data_without_missing = test_data[~missing_rows_mask] #only rows from test_data that dont have missing values
    preds_for_clean = model.predict(test_data_without_missing) #predicts the labels for rows that dont have missing values
    preds_for_missing = [most_common] * missing_rows_mask.sum() #predicts  labels for rows that have missing values

    combined_preds = np.empty(len(test_data), dtype=int) #combines for clean and missing rows
    combined_preds[~missing_rows_mask] = preds_for_clean 
    combined_preds[missing_rows_mask] = preds_for_missing

    return accuracy_score(test_labels, combined_preds), accuracy_score(test_labels[missing_rows_mask], preds_for_missing) ##accuracy for both test set and the rows with missing values


def drop_columns(X_train, X_test, y_train, y_test, missing_rows_mask):
    cols_to_drop = X_test.columns[X_test.isna().any()] #columns with missing values in test set
    model = RandomForestClassifier(n_estimators=200, max_depth=5, max_features='sqrt', min_samples_leaf=2, random_state=0)
    model.fit(X_train.drop(columns=cols_to_drop), y_train) #train the model on data without the columns that have missing values
    y_pred = model.predict(X_test.drop(columns=cols_to_drop))#predict on test set without the columns that have missing values
    return accuracy_score(y_test, y_pred), accuracy_score(y_test[missing_rows_mask], y_pred[missing_rows_mask]) #accuracy for both test set and the rows with missing values

def do_nothing(model, X_test, y_test, missing_rows_mask):
    y_pred = model.predict(X_test) #Predict on test set with NaNs as it is
    return accuracy_score(y_test, y_pred), accuracy_score(y_test[missing_rows_mask], y_pred[missing_rows_mask]) #accuracy for both test set and the rows with missing values

def mean_impute(model, X_train, X_test, y_test, missing_rows_mask):
    imputer = SimpleImputer(strategy='mean').fit(X_train) #fit mean imputer on training data
    X_test_filled = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns) #transform test data using the imputer fitted on training data
    y_pred = model.predict(X_test_filled) #predict on test set with mean imputed values
    return accuracy_score(y_test, y_pred), accuracy_score(y_test[missing_rows_mask], y_pred[missing_rows_mask]) #accuracy for both test set and the rows with missing values

def mult_impute(model, X_train, X_test, y_test, missing_rows_mask): 
    imputer = IterativeImputer(max_iter=5, random_state=0).fit(X_train) #fit iterative imputer on training data
    X_test_filled = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns) #transform test data using the imputer fitted on training data
    y_pred = model.predict(X_test_filled) #predict on test set with iterative imputed values
    return accuracy_score(y_test, y_pred), accuracy_score(y_test[missing_rows_mask], y_pred[missing_rows_mask]) #accuracy for both test set and the rows with missing values

def knn_impute(model, X_train, X_test, y_test, missing_rows_mask):
    imputer = KNNImputer().fit(X_train) #fit KNN imputer on training data
    X_test_filled = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns) #transform test data using the imputer fitted on training data
    y_pred = model.predict(X_test_filled) #predict on test set with KNN imputed values
    return accuracy_score(y_test, y_pred), accuracy_score(y_test[missing_rows_mask], y_pred[missing_rows_mask]) #accuracy for both test set and the rows with missing values

# ------------------ Main Evaluation Loop ----------------------

#store accuracies for each method and seed
all_results = defaultdict(list)

for seed in range(10):
    print(f"\n Running evaluation for seed {seed}")

    
    #(56-44)12 clean rows for test, rest for training
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean_rows, y_clean_rows, test_size=12, random_state=seed)

    #Combine 12 clean + 44 missing for final test set
    X_test_final = pd.concat([X_test_clean, X_missing_rows]).reset_index(drop=True)
    y_test_final = pd.concat([y_test_clean, y_missing_rows]).reset_index(drop=True)
    test_has_missing = X_test_final.isna().any(axis=1)

    #base RF model training on clean data
    base_model = RandomForestClassifier(n_estimators=200, max_depth=5, max_features='sqrt', min_samples_leaf=2, random_state=0)
    base_model.fit(X_train_clean, y_train_clean)

    #all methods and store all results
    all_results["Majority"].append(majority(base_model, X_test_final, y_test_final, y_train_clean, test_has_missing))
    all_results["Omit/Drop"].append(drop_columns(X_train_clean, X_test_final, y_train_clean, y_test_final, test_has_missing)) #retrain
    all_results["Do Nothing"].append(do_nothing(base_model, X_test_final, y_test_final, test_has_missing))
    all_results["Impute mean"].append(mean_impute(base_model, X_train_clean, X_test_final, y_test_final, test_has_missing))
    all_results["Multiple Imputation"].append(mult_impute(base_model, X_train_clean, X_test_final, y_test_final, test_has_missing))
    all_results["KNN Imputaion"].append(knn_impute(base_model, X_train_clean, X_test_final, y_test_final, test_has_missing))

# ------------------ Print Results ----------------------

print("\n Per-Seed Accuracy Breakdown")
for i in range(10): #loop through each seed
    print(f"\nSeed {i}:")
    for method, scores in all_results.items():
        print(f"  {method} - Full: {scores[i][0]:.4f}, Missing: {scores[i][1]:.4f}")

print("\n Summary: Average Accuracies Over 10 Splits")
for method, scores in all_results.items(): #avg accuracies for each method
    avg_full = np.mean([score[0] for score in scores])
    avg_missing = np.mean([score[1] for score in scores])
    print(f"{method} - Avg Full: {avg_full:.4f}, Avg Missing: {avg_missing:.4f}")

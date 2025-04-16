Name: Sanika Prashant Deshmukh
Course: ML Challenges in Real WOrld
Assignment: Elusive Exoplanets
Folder Contents: This folder contains three Python files for the exoplanet assignment, along with a findings document (PDF) and necessary CSV files.

Files in this Folder
1_data_profile.py

Generates basic summary of the dataset.Performs ydata-profiling for exploratory data analysis.Output is saved as an HTML report.

2_train_eval.py

Trains a Random Forest classifier with different missing value handling strategies.Compares 6 methods: majority fill, drop columns, predict-as-is, mean, iterative, and KNN imputation.Loops over 10 random seeds for robustness and prints detailed accuracy metrics.Recommendations are based on performance analysis.

3_blind_test.py

Trains a final model using all clean data.Applies the KNN Imputation strategy from analysis in file 2.Predicts the water class for unseen planets in the iac-atmospheres-blind.csv.

⚙️ Environment Setup
Python version: ✅ Works with Python ≤ 3.11

⚠️ ydata-profiling is not compatible with Python 3.13 or higher due to dependency limitations.

🧪 Dependencies
Install dependencies using:
pip install pandas numpy scikit-learn ydata-profiling 
RandomForestClassifier train_test_split SimpleImputer, IterativeImputer, KNNImputer, accuracy_score
If working in Colab, these are pre-installed. Just run:
!pip install ydata-profiling

Run Locally
python file_name.py 

📸 Screenshots
Included in findings.pdf to show ydata-profiling working on Google Colab (in case it fails locally due to Python version).

📝 Notes
All Python files assume iac-atmospheres.csv and iac-atmospheres-blind.csv are in the same directory.

Imports are handled at the top of each file.

Output from each run is printed in the terminal or Colab output.

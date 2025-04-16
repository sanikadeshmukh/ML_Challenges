# Sanika Prashant Deshmukh, date - 04-11-25, Assignment 1, Data profiling and visualization (works on colab (will attach screenshots in report) does not work on local if python version above 3.13)
import pandas as pd
from ydata_profiling import ProfileReport
data = pd.read_csv("iac-atmospheres.csv") #reading data from csv file
# Build the summary table
numeric_data = data.select_dtypes(include=["float64", "int64"]) #selecting only numeric columns
summary = pd.DataFrame({ #calculating summary statistics like min, max, mean, median and missing values
    "Min": numeric_data.min(),
    "Max": numeric_data.max(),
    "Mean": numeric_data.mean(),
    "Median": numeric_data.median(),
    "Missing Values": numeric_data.isnull().sum()
})

print(summary) #printing summary statistics



profile = ProfileReport(data, title="Profiling Report")
profile #shows visulation of data profiling report in colab, does not work on local if python version above 3.13
# profile.to_file("profiling_report.html") # saves the report as html file, does not work on local if python version above 3.13



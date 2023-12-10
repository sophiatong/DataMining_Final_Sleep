# DataMining_Final_Sleep
SI671 Final Project on Sleep Data

Dataset source: https://physionet.org/content/ucddb/1.0.0/
Data format: Tabular stationary data and time series data. 

Workflow:
EDA --> DTW for feature similarity --> NN model to predict the onset sleep apnea 

Gotchas:
1. need to change source code of edf reader package to read .edf files due to time stamp delimeter errors. 

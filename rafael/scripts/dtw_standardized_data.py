#!/usr/bin/python3
'''
Script to calculate standardized DTW data from patients.
'''
from concurrent.futures import ProcessPoolExecutor, wait
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from dtw import accelerated_dtw 
from functools import partial
from gc import collect
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# Define vars of interest and list of pairs
vars_of_interest = ['C3A2', 'abdo', 'Flow', 'Sum', 'Pulse', 'RightEye', 'C4A1', 'SpO2', 'BodyPos', 'ribcage', 'EMG', 'ECG', 'Lefteye']
var_pairs = list(combinations(vars_of_interest, 2))

# Manhattan distance function for DTW
manhattan_distance = lambda x, y: abs(x - y)

# Create results dir
if not os.path.isdir('DTW_standardized'):
    os.mkdir('DTW_standardized')

# Scaler
scaler = StandardScaler()

def calculate_dtw(var_1, var_2):
    '''
    Function to calculate the DTW distance between two variables
    '''
    d = accelerated_dtw(scaler.fit_transform(patient_data[var_1].to_numpy().reshape(-1, 1)), scaler.fit_transform(patient_data[var_2].to_numpy().reshape(-1,1)), dist=manhattan_distance)[0]
    return var_1, var_2, d

for patient, f in tqdm(enumerate(os.listdir('data'))):
    # Create data frame for storing the calculated results
    results = np.zeros((len(vars_of_interest), len(vars_of_interest)))
    results = pd.DataFrame(results)
    results.columns = vars_of_interest
    results.index = vars_of_interest

    # Read data
    patient_data = pd.read_parquet(f'data/{f}')

    # Remove repeated observations at the end
    patient_data = patient_data[patient_data['Pulse'] != patient_data['Pulse'].iloc[-1]]

    # Filter for the times the patient had pulse grater than 35, less than that is abnormal and was probably recorded while setting up the measuring device
    patient_data = patient_data[patient_data['Pulse'] > 35]

    # Down sample by taking the average of observations in windows of 20s. If no data is found in some window of time
    # bfill is used as a fill method
    patient_data.index = pd.to_datetime(patient_data['Time'], unit='s')
    patient_data = patient_data.resample('20s', axis=0).mean().bfill()
    
    print(patient_data.shape)

    # Instantiate multiprocessing executor
    with ProcessPoolExecutor(12) as executor:
        # Get DTW distances
        futures = [executor.submit(partial(calculate_dtw, i, j)) for i,j in var_pairs]

        # Wait for all calculations to finish
        done, not_done = wait(futures)

        if not_done:
            print('Some jobs did not finish')
            print(not_done)

        # Add to results df
        for job in done:
            i, j, d = job.result()
            results.loc[i,j] = d
            results.loc[j,i] = d

    # Save results
    results.to_csv(f'DTW_standardized/{patient}.csv')

    # # delete patient data to save memory
    del patient_data

    # Clean memory
    collect()

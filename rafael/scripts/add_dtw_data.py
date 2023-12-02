#!/usr/bin/python3
'''
Script to calculate DTW data from patients.
'''
from dtw import accelerated_dtw 
from gc import collect
from tqdm import tqdm
import pandas as pd
import os

# Create data frame for storing the calculated results
results = pd.DataFrame(columns=['Patient', 'DTW: Pulse - BodyPos', 'DTW: Pulse - ECG', 'DTW: Pulse - EMG', 'DTW: Pulse - SpO2'])

# Manhattan distance function for DTW
manhattan_distance = lambda x, y: abs(x - y)

for patient, f in tqdm(enumerate(os.listdir('data'))):
    # Read data
    patient_data = pd.read_parquet(f'data/{f}')

    # Remove repeated observations at the end
    patient_data = patient_data[patient_data['Pulse'] != patient_data['Pulse'].iloc[-1]]

    # Filter for the times the patient had pulse grater than 35, less than that is abnormal and was probably recorded while setting up the measuring device
    patient_data = patient_data[patient_data['Pulse'] > 35]

    # Down sample by taking the average of observations in windows of 2s. If no data is found in some window of time
    # bfill is used as a fill method
    patient_data.index = pd.to_datetime(patient_data['Time'], unit='s')
    patient_data = patient_data.resample('20s', axis=0).mean().bfill()

    # Get DTW distances
    d_BodyPos = accelerated_dtw(patient_data['Pulse'].to_numpy(), patient_data['BodyPos'].to_numpy(), dist=manhattan_distance)[0]
    d_ECG = accelerated_dtw(patient_data['Pulse'].to_numpy(), patient_data['ECG'].to_numpy(), dist=manhattan_distance)[0]
    d_EMG = accelerated_dtw(patient_data['Pulse'].to_numpy(), patient_data['EMG'].to_numpy(), dist=manhattan_distance)[0]
    d_Sp02 = accelerated_dtw(patient_data['Pulse'].to_numpy(), patient_data['SpO2'].to_numpy(), dist=manhattan_distance)[0]

    # Record results
    results.loc[len(results)] = [patient, d_BodyPos, d_ECG, d_EMG, d_Sp02]

    # Clean memory
    collect()

# Save results
results.to_csv('DTW_results.csv', index=False)

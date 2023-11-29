#!/usr/bin/python3
'''
Script to transform the .rec file of each patient into a CSV
'''

from tqdm import tqdm
import pandas as pd
import mne
import os

# Get EDF objects
data_objects = []
for f in os.listdir('data'):
    data_objects.append(mne.io.read_raw_edf(f'data/{f}', verbose='CRITICAL'))

# Make directory for storing the files if not already present
if not os.path.isdir('csv_rec_files'):
    os.mkdir('csv_rec_files')

# Iterate and save each of the files as CSV
for i, data_obj in tqdm(enumerate(data_objects)):
    vals = data_obj.get_data()
    vals = vals.T
    vals_df = pd.DataFrame(vals, columns = data_obj.ch_names, index = data_obj.times)
    vals_df.to_csv(f'csv_rec_files/patient_{i}_rec.csv')

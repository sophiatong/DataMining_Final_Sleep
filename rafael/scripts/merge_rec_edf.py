#!/usr/bin/python3
'''
Script to read and merge all .rec and .edf files.
'''
from gc import collect
from tqdm import tqdm
import pandas as pd
import mne
import os

# Obtain list of files
rec_dir = 'data_rec'
edf_dir = 'data_edf'
rec_files = sorted(os.listdir(rec_dir))
edf_files = sorted(os.listdir(edf_dir))

# Make merged dir if it's not already present
if not os.path.isdir('merged'):
    os.mkdir('merged')

# Merge each patient's .rec and .edf file
for i, pair in tqdm(enumerate(zip(rec_files, edf_files))):
    # Unpack
    rec_file, edf_file = pair

    # Get data objects
    rec_obj = mne.io.read_raw(f'{rec_dir}/{rec_file}', verbose = 'CRITICAL')
    edf_obj = mne.io.read_raw(f'{edf_dir}/{edf_file}', verbose = 'CRITICAL')

    # Get times for each dataset
    rec_time = rec_obj.times
    edf_time = edf_obj.times

    # Get data for each file
    rec_data = rec_obj.get_data()
    rec_data = rec_data.T
    edf_data = edf_obj.get_data()
    edf_data = edf_data.T

    # Create data frames
    rec_df = pd.DataFrame(rec_data, columns = rec_obj.ch_names)
    edf_df = pd.DataFrame(edf_data, columns = edf_obj.ch_names)

    # Add times to each data frame
    rec_df['Time'] = rec_obj.times
    edf_df['Time'] = edf_obj.times

    # Merge the data frames
    merged = pd.merge_asof(edf_df, rec_df)

    # Save to disk
    merged.to_parquet(f'merged/patient_{i}.gzip')

    # Clean memory
    collect()

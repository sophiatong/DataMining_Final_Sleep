#!/bin/bash

# Don't forget to specify the type of file to obtain as a command line argument
# example: ./read_files.sh edf

# Get the data
wget -r -A *.$1 -N -c -np https://physionet.org/files/ucddb/1.0.0

# Make data directory
mkdir -p data_$1

# Get data files into the current folder
mv physionet.org/files/ucddb/1.0.0/*.$1 data_$1

# rm download folder
rm -r physionet.org

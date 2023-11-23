#!/bin/bash

# Get the data
wget -r -A *.rec -N -c -np https://physionet.org/files/ucddb/1.0.0

# Make data directory
mkdir -p data

# Get data files into the current folder
mv physionet.org/files/ucddb/1.0.0/*.rec data

# rm download folder
rm -r physionet.org

# Change file extensions
for file in data/*.rec
do
	mv $file ${file%.rec}.edf
done

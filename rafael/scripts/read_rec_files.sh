#!/bin/bash

# Get the data
wget -r -A *.rec -N -c -np https://physionet.org/files/ucddb/1.0.0

# Get data files into the current folder
mv physionet.org/files/ucddb/1.0.0/*.rec .


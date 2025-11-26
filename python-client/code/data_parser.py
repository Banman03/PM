import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "data/"
available_files = os.listdir(base_dir)
file_data = np.ndarray((len(1, len(available_files))))

for i, file in enumerate(available_files):
    file_data[i] = pd.read_csv(file)


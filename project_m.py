import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# File path for the Excel file
file_path = 'C:\\Users\\lmueller\\Desktop\\Project M\\Project M Data.xlsx'

# Reading the data from the provided Excel file
# Adjust the sheet name if necessary
apple_data = pd.read_excel(file_path)

# Converting 'Date' to datetime and setting it as the index
apple_data['Date'] = pd.to_datetime(apple_data['Date'])
apple_data.set_index('Date', inplace=True)

# Ensure all relevant columns are treated as numeric
numeric_cols = ['Close/Last', 'Open', 'Daily High', 'Daily Low']
apple_data[numeric_cols] = apple_data[numeric_cols].apply(
    pd.to_numeric, errors='coerce')

# Calculating the 200-day moving average
apple_data['200d MA'] = apple_data['Close/Last'].rolling(window=200).mean()

# Displaying the beginning and end of the dataset
print(apple_data.head())
print(apple_data.tail())

# Exporting the dataset to an Excel file (if needed)
export_file_path = 'C:\\Users\\lmueller\\Desktop\\Project M\\Project M Data Processed.xlsx'
apple_data.to_excel(export_file_path)
